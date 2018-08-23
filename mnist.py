import math
import time
import argparse
import random
import os
from PIL import Image

import numpy as np
import torch
import torchvision

from models import MNISTModel, ConvMNISTModel, LSTMMNISTModel
from data_utils import KShotLoader, KShotData, split_array
from model_wrapper import MetaTrainWrapper
import tasks
from tasks import ClassifierTask
# from optim import SGD

import keras
from keras.datasets import mnist

import numpy as np

NUM_TEST_POINTS = 600
VALIDATION_SPLIT = [.8, .1, .1]

LINEAR = 0
CONV = 1
LSTM = 2

MODEL_TYPES = {'LINEAR': LINEAR, 'CONV': CONV, 'LSTM': LSTM}

def load_and_process_data(validation_split, n=5, k=1, metabatch_size=32, model_type=LINEAR):
	data = MNISTData(model_type)
	data.split(validation_split)
	loader = KShotLoader(data, n, k, metabatch_size)
	return loader

class MNISTData(KShotData):
	def __init__(self, model_type):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		if model_type == CONV:
			X = np.concatenate((x_train, x_test), axis=0).reshape(-1,1,28,28).astype('float32')
		else:
			X = np.concatenate((x_train, x_test), axis=0).reshape(-1,784).astype('float32')
		#normalize MNIST images
		X /= 255
		y = np.concatenate((y_train, y_test), axis=0)
		super(MNISTData, self).__init__(self.load_classes(X,y))

	def load_classes(self, x, y):
		self.input_size = x.shape[-1]
		n_classes = np.max(y)+1

		self.num_classes = n_classes
		classes_data = []
		for c in range(n_classes):
			classes_data.append(x[y==c])
		return classes_data

	def split(self, splits=[.8, .1, .1]):
		random.seed(1)
		random.shuffle(self.classes_data)

		self.train = []
		self.val = []
		self.test = []
		for c in self.classes_data:
			tr, va, te = split_array(c, splits)
			self.train.append(tr)
			self.val.append(va)
			self.test.append(te)

		print len(self.test[0])
		self.train = KShotData([torch.from_numpy(c).float().cuda() for c in self.train])
		self.val = KShotData([torch.from_numpy(c).float().cuda() for c in self.val])
		self.test = KShotData([torch.from_numpy(c).float().cuda() for c in self.test])
		del self.classes_data
		del self.class_idx

def wrap_model(model_type=LINEAR, n=5, lr=1e-4, finetune=1, inner_lr=.01, distributed=False, second_order=False):
	if model_type == CONV:
		model = ConvMNISTModel(n)
	elif model_type == LSTM:
		model = LSTMMNISTModel(n)
	else:
		model = MNISTModel(n)
	model.cuda()
	task = ClassifierTask()
	#TODO: Implement Task Map
	task_map = lambda x: task
	master_optim = torch.optim.Adam(model.parameters(), lr=lr)
	model = MetaTrainWrapper(model, task_map, finetune, inner_lr, master_optim, second_order=second_order, distributed=distributed)
	return model

def main(model_type=LINEAR, lr=1e-4, n=5, k=1, finetune=1, inner_lr=.4, second_order=False, metabatch_size=32, niters=40000, print_interval=100, eval_interval=500):
	assert model_type in MODEL_TYPES.values()
	validation_split = VALIDATION_SPLIT
	loader = load_and_process_data(validation_split, n, k, metabatch_size, model_type)

	module = wrap_model(model_type, n, lr=lr, finetune=finetune, inner_lr=inner_lr, second_order=second_order)
	module.train()
	train_loader = loader.train
	for i in range(niters):
		batch = next(train_loader)
		loss, metrics = module(batch)
		if (i+1) % print_interval == 0:
			module.log_history_point(i+1)
		if (i+1) % eval_interval == 0:
			module.eval()
			batch = next(loader.val)
			val_loss, val_metrics = module(batch)
			module.log_history_point(i+1)
			module.train()
		
	module.eval()
	for t, batch in enumerate(loader.test):
		module(batch)
		if t == NUM_TEST_POINTS-1:
			module.get_test_point()
			break
	module.plot()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='OmniGlot n-way k-shot Classifier')
	parser.add_argument('--model-type', default='LINEAR',
						help='one of [LINEAR, CONV, LSTM]')
	parser.add_argument('--niters', default=40000, type=int,
						help='Number of epochs to train. Default: 20000')
	parser.add_argument('--lr', default=1e-3, type=float,
						help='Learning rate to use. (used for meta optimizer in MAML). Default: 1e-3')
	parser.add_argument('--n', default=5, type=int,
						help='n-way. Default: 5')
	parser.add_argument('--k', default=1, type=int,
						help='k-shot. Default: 1')
	parser.add_argument('--ntasks', default=32, type=int,
						help='number of tasks to sample for MAML(metabatch_size). Default: 8')
	parser.add_argument('--nfinetune', default=1, type=int,
						help='number of finetuning steps in MAML. Default: 1')
	parser.add_argument('--inner-lr', default=.4, type=float,
						help='Learning rate for fine tune optimizer in MAML. Default: 1e-2')
	parser.add_argument('--second-order', action='store_true',
						help='use second order estimation for supervised MAML (instead of first order)')
	parser.add_argument('--print-interval', type=int, default=100,
						help='number of iterations between printing progress')
	parser.add_argument('--eval-interval', type=int, default=500,
						help='number of iterations between printing progress')
	args = parser.parse_args()


	main(MODEL_TYPES[args.model_type], args.lr, args.n, args.k, args.nfinetune, args.inner_lr,
		args.second_order, args.ntasks, args.niters, args.print_interval, args.eval_interval)