import torch
import torch.nn as nn
from torch.autograd import Variable

import matplotlib.pyplot as plt

from maml import MAML

class SumMeter(object):
	def __init__(self, val, n=-1):
		self.vals = [val.item()]
		self.n = float(n)
		self._mean = self.val

	def append(self, val):
		self.vals.append(val.item())
		if self.n > 0 and len(self.vals) > self.n:
			self.vals.pop(0)
		return self

	@property
	def mean(self):
		val_len = len(self.vals)
		if self.n <= 0:
			if val_len == 0:
				return self._mean
			rtn = self.val / val_len
			self.vals = []
			self._mean = rtn
			return rtn
		return self.val / min(self.n, val_len)

	@property
	def val(self):
		return sum(self.vals)

class WeightedSumMeter(object):
	def __init__(self, val, n=-1):
		self.vals = [[val[0].item()],[val[1].item()]]
		self.n = float(n)
		self._mean = self.val

	def append(self, val):
		self.vals[0].append(val[0].item())
		self.vals[1].append(val[1].item())
		if self.n > 0 and len(self.vals[0]) > self.n:
			self.vals[0].pop(0)
			self.vals[1].pop(0)
		return self

	@property
	def mean(self):
		rtn = self.val
		val_len = len(self.vals[0])
		if self.n <= 0:
			if val_len == 0:
				return self._mean
			rtn /= val_len
			self.vals = [[], []]
			self._mean = rtn
		else:
			rtn /= min(self.n, val_len)
		return rtn

	@property
	def val(self):
		return sum([v1/v2 for v1, v2 in zip(*self.vals)])

def init_sum_meters(metrics, n=-1):
	if not isinstance(metrics, (tuple, list)):
		return SumMeter(metrics, n)
	if len(metrics) == 2 and (list(filter(lambda x: isinstance(x, (tuple, list)), metrics)) == []):
		return WeightedSumMeter(metrics, n)
	return [init_sum_meters(metric, n) for metric in metrics]

def append_sum_meters(sum_meters, metrics):
	if not isinstance(sum_meters, (tuple, list)):
		sum_meters.append(metrics)
		return
	[append_sum_meters(meter, metrics[i]) for (i, meter) in enumerate(sum_meters)]

def get_mean(sum_meters):
	if not isinstance(sum_meters, (tuple, list)):
		return sum_meters.mean
	return [get_mean(meter) for meter in sum_meters]

def get_val(sum_meters):
	if not isinstance(sum_meters, (tuple, list)):
		return sum_meters.val
	return [get_val(meter) for meter in sum_meters]

class History(object):
	def __init__(self, val):
		self.vals = [val]

	def append(self, val):
		self.vals.append(val)

	@property
	def history(self):
		return self.vals

def init_histories(points):
	if not isinstance(points, (tuple, list)):
		return History(points)
	return [init_histories(point) for point in points]

def append_histories(histories, points):
	if not isinstance(histories, (tuple, list)):
		histories.append(points)
		return
	return [append_histories(history, points[i]) for (i, history) in enumerate(histories)]

def get_histories(histories):
	if isinstance(histories, History):
		return histories.history
	return [get_histories(history) for history in histories]

def _plot(loss_tr, loss_v, acc_tr, acc_v, tpr_tr, tpr_v, fpr_tr, fpr_v, outpath=None):#inner_lr,lr,lr_decay,N,K,layer_norm):
#	print (data1)
	plt.subplot(141)
	plt.plot(loss_tr,label='train cost')
	plt.plot(loss_v,label='val cost')
	plt.legend()
	plt.subplot(142)
	plt.plot(acc_tr,label='train acc')
	plt.plot(acc_v,label='val acc')
	plt.legend()
	plt.subplot(143)
	plt.plot(tpr_tr,label='train tpr')
	plt.plot(tpr_v,label='val tpr')
	plt.legend()
	plt.subplot(144)
	plt.plot(fpr_tr,label='train fpr')
	plt.plot(fpr_v,label='val fpr')
	plt.legend()
	if outpath is None:
		plt.show()
	else:
		plt.savefig(outpath)

class MetaTrainWrapper(nn.Module):
	def __init__(self, module, task_map, finetune=1, fine_optim=None, optim=None, second_order=False, distributed=False, world_size=1, rank=-1):
		super(MetaTrainWrapper, self).__init__()
		self.module = module
		self.task_map = task_map
		self.finetune = finetune
		self.fine_optim = fine_optim
		self.optim = optim
		self.distributed = distributed
		self.init_distributed(world_size, rank)
		self.meta_module = MAML(self.module, self.finetune, self.fine_optim, self.task_map, second_order=second_order)
		self.train_history = None
		self.train_meter = None
		self.val_history = None
		self.val_meter = None

	def train(self, mode=True):
		assert self.optim is not None
		return super(MetaTrainWrapper, self).train(mode)

	def update_lr(self, lr):
		for group in self.optim.param_groups:
			group['lr'] = lr

	def forward(self, batch):
		for group in self.optim.param_groups:
			for p in group['params']:
				if p.grad is not None:
					p.grad = p.grad.data.contiguous()
		self.optim.zero_grad()
		if self.training:
			loss, metrics = self.meta_module(batch)
			self.optim.step()
		else:
			with torch.no_grad():
				loss, metrics = self.meta_module(batch)

		loss = [l for l in loss]
		self.add_history(loss, metrics)
		return loss, metrics

	def add_history(self, loss, metrics):
		if self.distributed:
			#consider adding distributed loss aggregation
			pass
		if self.training:
			#something with train history
			if self.train_meter is None:
				self.train_meter = init_sum_meters((loss, metrics))
			else:
				append_sum_meters(self.train_meter, (loss, metrics))
		else:
			#something with val history
			print('val add history')
			if self.val_meter is None:
				self.val_meter = init_sum_meters((loss, metrics))
			else:
				append_sum_meters(self.val_meter, (loss, metrics))

	def log_history_point(self, point):
		train_point = get_mean(self.train_meter)
		if self.train_history is None:
			self.train_history = init_histories(train_point)
		else:
			append_histories(self.train_history, train_point)
		val_point = 0
		if self.val_meter is not None:
			val_point = get_mean(self.val_meter)
			if self.val_history is None:
				self.val_history = init_histories(val_point)
			else:
				append_histories(self.val_history, val_point)
		print('e%d:'%(point), 'train:', train_point, 'valid:', val_point)

	def get_test_point(self):
		test_point = get_mean(self.val_meter)
		print('test:', test_point)

	def plot(self, outpath=None):
		train_history = get_histories(self.train_history)
		val_history = get_histories(self.val_history)

		#losses
		train_loss = train_history[0][-1]
		val_loss = val_history[0][-1]

		# accuracies
		train_accuracies = train_history[1][-1][0]
		val_accuracies = val_history[1][-1][0]

		# tpr
		train_tpr = train_history[1][-1][1]
		val_tpr = val_history[1][-1][1]

		# fpr
		train_fpr = train_history[1][-1][-1]
		val_fpr = val_history[1][-1][-1]

		_plot(train_loss, val_loss, train_accuracies, val_accuracies,
			train_tpr, val_tpr, train_fpr, val_fpr, outpath=outpath)

	def init_distributed(self, world_size=1, rank=-1):
		if self.distributed:
			torch.distributed.init_process_group(backend='gloo', world_size=world_size,
												init_method='file://distributed.dpt', rank=rank)