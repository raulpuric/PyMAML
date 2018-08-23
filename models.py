import torch
import torch.nn as nn
from torch.autograd import Variable

class OmniGlotModel(nn.Module):
	def __init__(self, n_classes):
		super(OmniGlotModel, self).__init__()
		# self.conv1 = nn.Conv2d(3, 64, 3)

		self.net = nn.Sequential(
				nn.Conv2d(3, 64, 3),
				# nn.GroupNorm(8, 64, affine=False),
				# nn.GroupNorm(8, 64, affine=True),
				nn.BatchNorm2d(64, momentum=1, affine=True),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(2,2),
				nn.Conv2d(64,64,3),
				# nn.GroupNorm(8, 64, affine=False),
				# nn.GroupNorm(8, 64, affine=True),
				nn.BatchNorm2d(64, momentum=1, affine=True),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(2,2),
				nn.Conv2d(64,64,3),
				# nn.GroupNorm(8, 64, affine=False),
				# nn.GroupNorm(8, 64, affine=True),
				nn.BatchNorm2d(64, momentum=1, affine=True),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(2,2)
		)
		self.out = nn.Linear(64, n_classes)
		self.reset_parameters()

	def forward(self, input):
		feats = self.net(input)
		return self.out(feats.view(input.size(0), -1).contiguous())

	def zero_grad(self, inplace=True):
		for n, p in super(OmniGlotModel, self).named_parameters():
			if p.grad is not None:
				if inplace:
					p.grad.detach_()
					p.grad.zero_()
				else:
					is_cuda = p.grad.is_cuda
					p.grad = Variable(torch.zeros_like(p.grad.data))
					if is_cuda:
						p.grad = p.grad.cuda()

	def reset_parameters(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				torch.nn.init.xavier_normal(m.weight)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				torch.nn.init.xavier_normal(m.weight)
				if m.bias is not None:
					m.bias.data.fill_(1)
					

class MNISTModel(nn.Module):
	def __init__(self, n_classes):
		super(MNISTModel, self).__init__()

		self.net = nn.Sequential(
				nn.Linear(784,256),
				nn.ReLU(inplace=True),
				nn.Linear(256,256),
				nn.ReLU(inplace=True),
				nn.Linear(256, n_classes)
		)

	def forward(self, input):
		return self.net(input)


class ConvMNISTModel(nn.Module):
	def __init__(self, n_classes):
		super(ConvMNISTModel, self).__init__()

		self.net = nn.Sequential(
				nn.Conv2d(1, 32, 3),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(2,2),
				nn.Conv2d(32,64,3),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(2,2),
				nn.Dropout(.75),
		)
		self.fc1 = nn.Linear(1600, 128)
		self.dropout = nn.Dropout(.5)
		self.out = nn.Linear(128, n_classes)

	def forward(self, input):
		feats = self.net(input)
		feat_vec = self.dropout(self.fc1(feats.view(input.size(0), -1).contiguous()))

		return self.out(feat_vec)


class MyLSTMCell(nn.Module):
	def __init__(self, dim_in, dim_hid):
		super(MyLSTMCell, self).__init__()
		self.gating = nn.Linear(dim_in+dim_hid, 4*dim_hid)

	def forward(self, inputs, hidden):
		h, c = hidden
		gates = self.gating(torch.cat([inputs, h], 1))
		i, j, f, o = torch.chunk(gates, 4, 1)
		new_c = c * torch.sigmoid(f) + torch.sigmoid(i) * torch.tanh(j)
		new_h = torch.tanh(new_c) * torch.sigmoid(o)
		return new_h, new_c


class LSTMMNISTModel(nn.Module):
	def __init__(self, n_classes):
		super(LSTMMNISTModel, self).__init__()
		# self.conv1 = nn.Conv2d(3, 64, 3)
		self.n_hid = 64
		self.row_size = 28
		self.rnn = MyLSTMCell(self.row_size, self.n_hid)
		self.out = nn.Linear(self.n_hid, n_classes)

	def forward(self, input):
		hidden = (torch.zeros(input.size(0), self.n_hid), torch.zeros(input.size(0), self.n_hid))
		if input.is_cuda:
			hidden = [h.cuda() for h in hidden]
		input = input.view(input.size(0), -1, self.row_size).contiguous()
		for t in range(input.size(1)):
			hidden = self.rnn(input[:,t], hidden)
		return self.out(hidden[0])