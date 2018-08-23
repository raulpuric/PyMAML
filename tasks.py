from task_utils import TaskModule
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from eval_funcs import check_acc

SOFTMAX_CLF = 0
SIGMOID_CLF = 1

CLF_TYPES = [SOFTMAX_CLF, SIGMOID_CLF]

def make_onehot(outs, targets):
	if outs.size(-1) == 1:
		return targets.view(targets.size()+torch.Size([1])).float()
	onehot = torch.zeros(outs.size()).type_as(targets[1].data)
	dim = len(targets.size())
	index = targets.data.view(-1,1).contiguous()
	onehot.scatter_(dim, index.long(), 1)
	onehot = Variable(onehot)
	return onehot.float()

class ClassifierTask(TaskModule):
	def __init__(self, clf_type=SOFTMAX_CLF, forward_fn=None, criterion_fn=None, metric_fn=None):
		self.clf_type = clf_type
		assert self.clf_type in CLF_TYPES
		if forward_fn is None:
			forward_fn = self.forward_fn
		if criterion_fn is None:
			criterion_fn = self.criterion_fn
		if metric_fn is None:
			metric_fn = self.metric_fn
		super(ClassifierTask,self).__init__(forward_fn, criterion_fn, metric_fn)
		if self.clf_type == SOFTMAX_CLF:
			self.loss_fn = torch.nn.CrossEntropyLoss()
		else:
			self.loss_fn = torch.nn.BCEWithLogitsLoss()

	def forward(self, module, *inputs, **kwargs):
		rtn = super(ClassifierTask,self).forward(module, *inputs, **kwargs)
		if hasattr(self, 'onehot_targets_'):
			del self.onehot_targets_
		return rtn

	def forward_fn(self, module, *inputs, **kwargs):
		return module(inputs[0], **kwargs)

	def criterion_fn(self, outs, *inputs, **kwargs):
		if self.clf_type == SOFTMAX_CLF:
			# print (torch.max(outs,1)[1], inputs[1])
			return [self.loss_fn(outs, inputs[1])]
		else:
			return [self.loss_fn(outs, inputs[1])]

	def metric_fn(self, outs, criterion_metrics, *inputs, **kwargs):
		acc, (tp, fp) = check_acc(outs, inputs[1])
		tp = (torch.sum(tp[0]), torch.sum(tp[1]))
		fp = (torch.sum(fp[0]), torch.sum(fp[1]))
		return [acc, tp, fp] 