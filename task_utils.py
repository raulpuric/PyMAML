from collections import OrderedDict

import torch
import torch.nn as nn

def param_dict(self, destination=None, prefix='', clone=False):
	if destination is None:
		destination = OrderedDict()
	for name, param in self._parameters.items():
		if param is not None:
			destination[prefix + name] = param.detach().data.clone() if clone else param.data
	for name, module in self._modules.items():
		if module is not None:
			param_dict(module, destination, prefix + name + '.', clone=clone)
	return destination

def load_param_dict(module, copy, strict=True):
	own_state = param_dict(module)
	for name, param in copy.items():
		if name in own_state:
			try:
				own_state[name].copy_(param)
			except Exception:
				raise RuntimeError('While copying the parameter named {}, '
									'whose dimensions in the model are {} and '
									'whose dimensions in the checkpoint are {}.'
									.format(name, own_state[name].size(), param.size()))
		elif strict:
			raise KeyError('unexpected key "{}" in state_dict'
							.format(name))
	if strict:
		missing = set(own_state.keys()) - set(copy.keys())
		if len(missing) > 0:
			raise KeyError('missing keys in state_dict: "{}"'.format(missing))

def grad_dict(self, destination=None, prefix='', clone=False):
	if destination is None:
		destination = OrderedDict()
	for name, param in self._parameters.items():
		if param is not None:
			if param.grad is not None:
				destination[prefix + name] = param.grad.detach().data.clone() if clone else param.grad.data
			else:
				destination[prefix + name] = None
	for name, module in self._modules.items():
		if module is not None:
			grad_dict(module, destination=destination, prefix=prefix+name+'.', clone=clone)
	return destination

def update_grad_dict(module, grad_state_dict, strict=True, accumulate=True):
	own_state = grad_dict(module)
	for name, grad in own_state.items():
		if grad is None:
			continue
		if name in grad_state_dict:
			grad2update = grad_state_dict[name]
			try:
				if grad2update is None:
					if grad is None:
						continue
					grad_state_dict[name] = grad.clone()
				else:
					if accumulate:
						grad2update.add_(grad)
					else:
						grad2update.copy_(grad)
			except Exception:
				raise RuntimeError('While copying the parameter named {}, '
									'whose dimensions in the fine tuned model are {} and '
									'whose dimensions in the meta model are {}.'
									.format(name, grad.size(), grad_state_dict[name].size()))
		elif strict:
			raise KeyError('unexpected key "{}" in fine tuned model'
							.format(name))

def load_grad_dict(module, gradient_dict, strict=True):
	own_state = grad_dict(module)
	for name, grad in gradient_dict.items():
		if grad is None:
			continue
		if name in own_state:
			try:
				grad2update = own_state[name]
				if grad2update is None:
					continue
				else:
					grad2update.copy_(grad)

			except Exception:
				raise RuntimeError('While copying the parameter named {}, '
									'whose dimensions in the model are {} and '
									'whose dimensions in the checkpoint are {}.'
									.format(name, own_state[name].size(), grad.size()))
		elif strict:
			raise KeyError('unexpected key "{}" in gradient_dict'
							.format(name))
	if strict:
		missing = set(own_state.keys()) - set(gradient_dict.keys())
		if len(missing) > 0:
			raise KeyError('missing keys in gradient_dict: "{}"'.format(missing))

class TaskModule(nn.Module):
	def __init__(self, forward_fn=None, criterion_fn=None, metric_fn=None):
		super(TaskModule, self).__init__()
		self.forward_fn = forward_fn
		self.criterion_fn = criterion_fn
		self.metric_fn = metric_fn

	def forward(self, module, *inputs, **kwargs):
		if self.forward_fn is None:
			outs = module(*inputs, **kwargs)
		else:
			outs = self.forward_fn(module, *inputs, **kwargs)

		criterion_metrics = []
		if self.criterion_fn is not None:
			criterion_metrics = self.criterion_fn(outs, *inputs, **kwargs)

		other_metrics = []
		if self.metric_fn is not None:
			other_metrics = self.metric_fn(outs, criterion_metrics, *inputs, **kwargs)

		return outs, criterion_metrics, other_metrics

	def initialize(self, task_id, module):
		pass