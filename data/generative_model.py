import numpy as np
from torch.utils.data import Dataset
import torch


class GenerativeModelDataset(Dataset):
	def __init__(self, x):
		self.n = np.shape(x)[0]
		self.feature = x

	def __len__(self):
		return self.n

	def __getitem__(self, idx):
		return torch.tensor(self.feature[idx, :], dtype=torch.float32)


class NonlinearFactorModelSingleIndex:
	def __init__(self, p, r, index_func, sigma=1.0):
		self.p = p
		self.r = r
		self.beta_matrix = np.reshape(np.random.uniform(-1, 1, p * r), (r, p))
		self.index_func = index_func
		self.sigma = sigma

	def sample(self, z):
		n = np.shape(z)[0]
		f_logit = np.matmul(z, self.beta_matrix)
		part_f = self.index_func(f_logit)
		part_u = np.reshape(np.random.uniform(-self.sigma, self.sigma, n * self.p), (n, self.p))
		x = part_f + part_u
		return x


