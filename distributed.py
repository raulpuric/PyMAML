import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist

warn_on_half = True

def broadcast_params(params):
	for p in params:
		if torch.is_tensor(p):
			torch.distributed.broadcast(p, 0)

def reduce_gradients(module):
	buckets = {}
	for name, param in module.named_parameters():
		if param.requires_grad and param.grad is not None:
			tp = type(param.data)
			if tp not in buckets:
				buckets[tp] = []
			buckets[tp].append(param)
	if warn_on_half:
		if torch.cuda.HalfTensor in buckets:
			print("WARNING: gloo dist backend for half parameters may be slow." +
				  " It is recommended to use the NCCL backend in this case.")
			warn_on_half = False

	for tp in buckets:
		bucket = buckets[tp]
		grads = [param.grad.data for param in bucket]
		coalesced = _flatten_dense_tensors(grads)
		dist.all_reduce(coalesced)
		torch.cuda.synchronize()
		coalesced /= dist.get_world_size()
		for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
			buf.copy_(synced)
