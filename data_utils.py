from operator import itemgetter
import random

import numpy as np

import torch
from torch.autograd import Variable

class TaskBatch(object):
	def __init__(self, task_batch, size):
		self.task_batch = task_batch
		self.size = size
	def __len__(self):
		return self.size

	def __iter__(self):
		return self.task_batch

def sample_ids(X, size):
	return np.random.choice(len(X), size)

def get_sample_from_ids(X, Y, ids):
	_X = X[ids].contiguous()
	_Y = Y[ids].contiguous()
	return Variable(_X, requires_grad=False), Variable(_Y, requires_grad=False)

def split_array(array, splits=[-1]):
	"""splits an array into `split` proportions. `split` must sum to 1."""
	array_len = len(array)
	inds = list(range(array_len))
	random.shuffle(inds)
	rtn = []
	ctr = 0
	def get_split(arr, idx):
		if isinstance(arr, np.ndarray):
			return arr[idx]
		else:
			return itemgetter(*idx)(arr)
	for split in splits:
		if split < 0:
			idx=inds[ctr:]
			rtn.append(get_split(array,idx))
			return rtn
		if split < 1.:
			split = int(array_len*split)
		idx = inds[ctr:ctr+split]
		rtn.append(get_split(array,idx))
		ctr += split
	return rtn 

class KShotData(object):
	def __init__(self, classes_data):
		self.classes_data = classes_data
		self.num_classes = len(classes_data)
		self.class_idx = [range(len(x)) for x in self.classes_data]

	def get_class(self, cls, n=1):
		ids = random.sample(self.class_idx[cls], n)
		cls = self.classes_data[cls]
		return cls[ids]

	def split(self, splits=[.8, .1, .1]):
		rtn = split_array(self.classes_data, splits)
		return [KShotData(data) for data in rtn]

class KShotLoader:
	def __init__(self, kshotdata, n, k, metabatch_size, transform=None):
		self.data = kshotdata
		self.n = n
		self.k = k
		self.metabatch_size = metabatch_size
		self.transform = transform

	@property
	def train(self):
		data = self.data
		if hasattr(data, 'train'):
			data = data.train
		while True:
			yield TaskBatch(self.sample_tasks(data), self.metabatch_size)

	@property
	def val(self):
		data = self.data
		if hasattr(data, 'val'):
			data = data.val
		while True:
			yield TaskBatch(self.sample_tasks(data), self.metabatch_size)

	@property
	def test(self):
		data = self.data
		if hasattr(data, 'test'):
			data = data.test
		while True:
			yield TaskBatch(self.sample_tasks(data), self.metabatch_size)

	def sample_tasks(self, data):
		"""returns task description in the format [task_id, finetune data, eval data]"""
		for _ in range(self.metabatch_size):
			tr_data, tr_targets, sampled_classes, class_map = self.sample_task_data(data)
			yield 0, [(tr_data, tr_targets)], [self.sample_task_data(data, 1, sampled_classes, class_map)[:2]]

	def sample_task_data(self, data, k=None, sampled_classes=None, class_map=None):
		if k is None:
			k = self.k
		if sampled_classes is None:
			sampled_classes = classes = random.sample(xrange(data.num_classes), self.n)
		else:
			classes = sampled_classes
		if class_map is None:
			class_map = {c: i for i, c in enumerate(sampled_classes)}
		sampled_class_data = torch.cat([data.get_class(c, k) for c in classes])
		if self.transform is not None:
			sampled_class_data = self.transform(sampled_class_data)
		if k > 1:
			classes = [c for c in classes for _ in range(k)]
		sampled_class_data = Variable(sampled_class_data, requires_grad=True)
		targets = Variable(torch.LongTensor([class_map[c] for c in classes]), requires_grad=True)
		if sampled_class_data.is_cuda:
			targets = targets.cuda()
		return sampled_class_data, targets, sampled_classes, class_map

