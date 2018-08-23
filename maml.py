import torch
import torch.nn as nn

from task_utils import param_dict, load_param_dict, TaskModule
from task_utils import grad_dict, update_grad_dict, load_grad_dict
from distributed import reduce_gradients

def normal_backward(x, variables=None, retain_graph=None, create_graph=False):
	return x.backward(variables, retain_graph, create_graph)

def transpose_list(list_of_lists):
	return [tuple(zip(*x)) if isinstance(x[0], tuple) else list(x) for x in zip(*list_of_lists)]


class GradientTaskModule(nn.Module):
	def __init__(self):
		super(GradientTaskModule, self).__init__()

	def forward(self, task, module, input_loader, params_for_grad, compute_grad=True, create_graph=False, gradient_average=1):
		outlist = []
		rtn_grads = [None] * len(params_for_grad)
		for i, input in enumerate(input_loader):
			outs, criterions, metrics = task(module, *input)
			criterion = criterions[0]
			if compute_grad:
				grads = torch.autograd.grad(criterion/gradient_average, params_for_grad, create_graph=create_graph)
				if i == 0:
					rtn_grads = grads
				else:
					rtn_grads = [g1+g2 for g1, g2 in zip(rtn_grads, grads)]
				del grads
		return rtn_grads, (outs, criterion, metrics)

def reduce_sum(tens):
	t_sum = 0
	for t in tens:
		if len(t.size()) >= 1:
			if len(t.size()) > 1 or t.size(0) != 0:
				t = t.sum()
		t_sum += t
	return t_sum

def mean(lst):
	length = float(len(lst))
	return reduce_sum(lst)/length

def grad_step_params(params, grads, lr, inplace=False):
	if inplace:
		for p, g in zip(params, grads):
			if g is None:
				continue
			p.data.add_(-lr*g.data)
		return 
	return [p - lr*g if g is not None else p for (p, g) in zip(params, grads)]

def set_grads_for_params(params, grads):
	for p, g in zip(params, grads):
		if g is None:
			continue
		p.grad = g.contiguous()

def get_params_for_grad(module, destination=None):
	if destination is None:
		destination = []
	for name, param in module._parameters.items():
		if param is not None:
			destination.append(param)
	for name, module in module._modules.items():
		if module is not None:
			get_params_for_grad(module, destination)
	return destination

def set_params_with_grad(module, params):
	for name, param in module._parameters.items():
		if param is None:
			continue
		module._parameters[name] = params.pop(0)
	for name, module in module._modules.items():
		if module is not None:
			set_params_with_grad(module, params)

class MAML(nn.Module):
	def __init__(self, module, num_finetune, inner_lr, task_map, second_order=False, maml_task=None, distributed=False):
		super(MAML, self).__init__()
		self.module = module
		self.num_finetune = num_finetune
		self.inner_lr = inner_lr
		self.task_map = task_map
		self.second_order = second_order
		if maml_task is None:
			maml_task = GradientTaskModule()
		self.maml_task = maml_task
		self.distributed = distributed
		if self.distributed:
			for p in self.module.state_dict().values():
				if torch.is_tensor(p):
					torch.distributed.broadcast(p, 0)

	def forward(self, task_batch):
		loss_start = 0
		loss_end = 0
		meta_eval_loss = 0
		metrics_start = []
		metrics_end = []
		meta_metrics = []

		meta_grad_dict = grad_dict(self.module, clone=True)
		meta_param_dict = param_dict(self.module, clone=True)

		metabatch_size = len(task_batch)
		self.metabatch_size = metabatch_size

		for i, (task_id, loader, val_loader) in enumerate(task_batch):
			load_param_dict(self.module, meta_param_dict)
			params_for_grad = get_params_for_grad(self.module)

			task = self.task_map(task_id)
			task.initialize(task_id, self.module)

			losses, metrics, tgrads = self.finetune(task, loader, params_for_grad)

			loss_start += losses[0]
			loss_end += losses[1]
			if metrics_start is None:
				metrics_start = []
				metrics_end = []
			metrics_start.append(metrics[0])
			metrics_end.append(metrics[1])

			mloss, mmetrics = self.meta_eval(task, val_loader, params_for_grad)
			meta_eval_loss += mloss
			meta_metrics.append(mmetrics)

			update_grad_dict(self.module, meta_grad_dict)

		loss_start /= metabatch_size
		loss_end /= metabatch_size
		meta_eval_loss /= metabatch_size

		metrics_start = [(reduce_sum(metric[0]), reduce_sum(metric[1])) if isinstance(metric,tuple) else mean(metric) for metric in transpose_list(metrics_start)]
		metrics_end = [(reduce_sum(metric[0]), reduce_sum(metric[1])) if isinstance(metric,tuple) else mean(metric) for metric in transpose_list(metrics_end)]
		meta_metrics = [(reduce_sum(metric[0]), reduce_sum(metric[1])) if isinstance(metric,tuple) else mean(metric) for metric in transpose_list(meta_metrics)]

		self.synchronize()

		return (loss_start, loss_end, meta_eval_loss), (metrics_start, metrics_end, meta_metrics)

	def finetune(self, task, loader, params_for_grad):
		loss_start = None
		loss_end = 0
		metrics_start = None
		metrics_end = None
		training_status = self.training
		self.module.train()
		for i in range(self.num_finetune):
			with torch.enable_grad():
				grads, (_, loss_end, metrics_end) = self.maml_task(task, self.module, loader, params_for_grad, create_graph=self.second_order and self.training)
				new_params = grad_step_params(params_for_grad, grads, self.inner_lr, inplace=False)
			params_for_grad = list(new_params)
			set_params_with_grad(self.module, new_params)
			if loss_start is None:
				loss_start = loss_end
				metrics_start = metrics_end
		self.module.train(training_status)
		return (loss_start, loss_end), (metrics_start, metrics_end), grads

	def meta_eval(self, task, loader, params_for_grad):
		meta_loss = 0
		meta_metrics = None
		grads, (_, meta_loss, meta_metrics) = self.maml_task(task, self.module, loader, params_for_grad if self.second_order else get_params_for_grad(self.module), compute_grad=self.training, gradient_average=self.metabatch_size)
		set_grads_for_params(params_for_grad, grads)
		set_params_with_grad(self.module, params_for_grad)

		return meta_loss, meta_metrics

	def synchronize(self):
		if self.distributed:
			reduce_gradients(self.module)
