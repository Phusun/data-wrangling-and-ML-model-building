import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer

class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels

		# Make self.seqs as a List of which each element represent visits of a patient
		# by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# self.seqs = [i for i in range(len(labels))]  # replace this with your implementation.

		self.seqs = seqs

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
	where N is minibatch size, seq_i is a Numpy (or Scipy Sparse) array, and label is an int value

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	# Return the following two things
	# 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
	# 2. Tensor contains the label of each sequence

	N = len(batch)
	length_label_M = [(x[0].shape[0], x[1], x[0]) for x in batch]
	sorted_by_length = sorted(length_label_M, key=lambda tup: tup[0], reverse=True)

	lengths_tensor = torch.LongTensor([x[0] for x in sorted_by_length])
	labels_tensor = torch.LongTensor([x[1] for x in sorted_by_length])
	seqs_M = [x[2].toarray() for x in sorted_by_length]	# if sparse format
	# seqs_M = [x[2] for x in sorted_by_length]    # if dense format

	num_feats = seqs_M[0].shape[1]

	max_visits = int(lengths_tensor[0])

	seqs_zeropadded = [np.vstack((x, np.zeros((max_visits-x.shape[0], num_feats)))) for x in seqs_M]

	seqs_stacked = np.stack(seqs_zeropadded, axis=0)

	seqs_tensor = torch.FloatTensor(seqs_stacked)

	return (seqs_tensor, lengths_tensor), labels_tensor
