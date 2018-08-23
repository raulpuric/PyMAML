import math
import time
import argparse
import random
import os
from PIL import Image

import numpy as np
import torch
import torchvision

from models import OmniGlotModel
from data_utils import KShotLoader, KShotData
from model_wrapper import MetaTrainWrapper
import tasks
from tasks import ClassifierTask
# from optim import SGD


NUM_TEST_POINTS = 600
VALIDATION_SPLIT = [1100, 100, -1]


class Normalize(object):
	def __init__(self, mean, std):
		self.mean = torch.FloatTensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
		self.std = torch.FloatTensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
	def __call__(self, inputs):
		return inputs.sub_(self.mean).div_(self.std)


def load_and_process_data(path, validation_split, n=5, k=1, metabatch_size=32):
	data = OmniGlotData(path)
	data.split(validation_split)
	loader = KShotLoader(data, n, k, metabatch_size, transform=Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426]))
	return loader

#TODO: Implement KShotData, OmniGlotData, MiniImageNetData, KShotLoader
class OmniGlotData(KShotData):
	def __init__(self, path):
		self.path = path
		super(OmniGlotData, self).__init__(self.load_classes())

	def get_char_folders(self):
		folders = [os.path.join(self.path, family, character) \
			for family in os.listdir(self.path) \
			if os.path.isdir(os.path.join(self.path, family)) \
			for character in os.listdir(os.path.join(self.path, family))]
		return folders 

	def load_classes(self):
		folders = self.get_char_folders()
		classes = []
		for folder in folders:
			classes.append(self.folder_to_tensor(folder))
		return classes

	def folder_to_tensor(self, folder):
		files = [os.path.join(folder, f) for f in os.listdir(folder)]
		return torch.stack([self.img_from_path(f) for f in files]).cuda()

	def img_from_path(self, f):
		im = Image.open(f).convert('RGB').resize((28, 28), resample=Image.LANCZOS)
		return torch.from_numpy(np.array(im).transpose(2, 0, 1)/255).float()
		# return torch.from_numpy(1 - (np.array(im).transpose(2, 0, 1)/255.)).float()

	def split(self, splits=[1100, 100, -1]):
		random.seed(1)
		self.train, self.val, self.test = super(OmniGlotData, self).split(splits)
		del self.classes_data
		del self.class_idx

def wrap_model(n=5, lr=1e-4, finetune=1, inner_lr=.01, distributed=False, second_order=False):
	model = OmniGlotModel(n)
	model.cuda()
	task = ClassifierTask()
	task_map = lambda x: task
	master_optim = torch.optim.Adam(model.parameters(), lr=lr)
	model = MetaTrainWrapper(model, task_map, finetune, inner_lr, master_optim, second_order=second_order, distributed=distributed)
	return model


def main(data_path, lr=1e-4, n=5, k=1, finetune=1, inner_lr=.4, second_order=False, metabatch_size=32, niters=40000, print_interval=100, eval_interval=500):
	validation_split = VALIDATION_SPLIT
	loader = load_and_process_data(data_path, validation_split, n, k, metabatch_size)
	
	module = wrap_model(n=n, lr=lr, finetune=finetune, inner_lr=inner_lr, second_order=second_order)

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
	parser.add_argument('--data-path', default='./data/omniglot',
						help='path where data is located. (required)')
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


	main(args.data_path, args.lr, args.n, args.k, args.nfinetune, args.inner_lr,
		args.second_order, args.ntasks, args.niters, args.print_interval, args.eval_interval)