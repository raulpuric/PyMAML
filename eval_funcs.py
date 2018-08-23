import torch

def get_counts(ys,n_classes):
	batch_size = ys.size(0)
	if n_classes==1:
		count=torch.sum(ys).float().data
		return count, batch_size-count
	y_onehot = torch.zeros(batch_size, n_classes).float()
	if ys.is_cuda:
		y_onehot=y_onehot.cuda()
	y_onehot.scatter_(1, ys.data.view(-1, 1), 1.)
	counts = y_onehot.sum(dim=0).squeeze()
	not_counts = batch_size - counts
	return counts, not_counts

def get_correct_clf(outputs,y_true):
	return torch.eq(outputs,y_true).float()


def get_tpr_fpr(outputs,y_true,n_class):
	counts, not_counts = get_counts(y_true.long(),n_class)

	correct = get_correct_clf(outputs,y_true)
	wrong = 1-correct

	tp=torch.zeros(n_class)
	fp=torch.zeros(n_class)
	if correct.is_cuda:
		tp=tp.cuda()
		fp=fp.cuda()

	class_iter = range(n_class)
	if n_class == 1:
		class_iter = [1]

	for c in class_iter:
		class_idx = torch.eq(y_true,c)
		class_outputs = correct[class_idx]

		out_pred_idx = torch.eq(outputs,c)
		false_class_outputs = wrong[out_pred_idx]

		if n_class == 1:
			c = 0

		tp[c] = torch.sum(class_outputs)
		fp[c] = torch.sum(false_class_outputs)

	return (tp, counts), (fp, not_counts)


def check_acc(outs,ys,_max=True):
	n_class=outs.size(1)
	if _max and n_class != 1:
		outputs=outs.max(1)[1]
	else:
		outputs=(torch.nn.functional.sigmoid(outs)>.5).view(-1)
	y_true=ys
	if len(y_true.size())!=1:
		y_true=torch.max(y_true,1)[1]
	outputs=outputs.type_as(y_true)
	acc=torch.mean(get_correct_clf(outputs,y_true))
	return acc.type_as(outs), get_tpr_fpr(outputs, y_true, n_class)