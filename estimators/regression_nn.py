import torch
import numpy as np
from torch import nn
from collections import OrderedDict
from torch.utils.data import DataLoader

import sys
sys.path.append("..")
sys.path.append(".")
from data.regression import RegressionDataset


class RegressionNN(nn.Module):
	def __init__(self, d, depth, width, input_dropout=False, dropout_rate=0.0):
		super(RegressionNN, self).__init__()
		self.use_input_dropout = input_dropout
		self.input_dropout = nn.Dropout(p=dropout_rate)

		relu_nn = [('linear1', nn.Linear(d, width)), ('relu1', nn.ReLU())]
		for i in range(depth - 1):
			relu_nn.append(('linear{}'.format(i+2), nn.Linear(width, width)))
			relu_nn.append(('relu{}'.format(i+2), nn.ReLU()))
		relu_nn.append(('linear{}'.format(depth+1), nn.Linear(width, 1)))

		self.relu_stack = nn.Sequential(
			OrderedDict(relu_nn)
		)

	def forward(self, x, is_training=False):
		if self.use_input_dropout and is_training:
			x = self.input_dropout(x)
		pred = self.relu_stack(x)
		return pred


def train_loop(data_loader, model, loss_fn, optimizer):
	loss_rec = {'l2_loss': 0.0}
	for batch, (x, y) in enumerate(data_loader):
		pred = model(x, is_training=True)
		loss = loss_fn(pred, y)
		loss_rec['l2_loss'] += loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	loss_rec['l2_loss'] /= len(data_loader)
	return loss_rec


def test_loop(data_loader, model, loss_fn):
	loss_sum = 0
	with torch.no_grad():
		for x, y in data_loader:
			pred = model(x, is_training=False)
			loss_sum += loss_fn(pred, y).item()
	loss_rec = {'l2_loss': loss_sum / len(data_loader)}
	return loss_rec


class RegressionNNEstimator:
	def __init__(self, dim_x=4, depth=3, width=32, num_epoch=200, learing_rate=1e-3, batch_size=64):
		self.model = True
		self.learning_rate = learing_rate
		self.num_epoch = num_epoch
		self.depth = depth
		self.width = width
		self.p = dim_x
		self.batch_size = batch_size

	def single_fit_and_predict(self, train_data_loader, valid_data_loader, test_x):
		device = "cuda" if torch.cuda.is_available() else "cpu"
		nn_model = \
			RegressionNN(d=self.p, depth=self.depth, width=self.width).to(device)
		mse_loss = nn.MSELoss()
		optimizer = torch.optim.Adam(nn_model.parameters(), lr=self.learning_rate)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

		cur_valid = 1e9
		last_update = 1e9
		for epoch in range(self.num_epoch):
			train_losses = train_loop(train_data_loader, nn_model, mse_loss, optimizer)
			scheduler.step()
			valid_losses = test_loop(valid_data_loader, nn_model, mse_loss)
			if valid_losses['l2_loss'] < cur_valid:
				cur_valid = valid_losses['l2_loss']
				last_update = epoch
				with torch.no_grad():
					test_y = nn_model(torch.tensor(test_x, dtype=torch.float32)).detach().numpy()
		print(f'[RegressionNN] last_update = {last_update}, valid loss = {cur_valid}')
		return cur_valid, test_y

	def model_fit_and_predict(self, x, y, valid_x, valid_y, test_x):
		# x shape = [n, p]
		# y shape = [n]
		y_ex = np.reshape(y, (np.shape(y)[0], 1))
		valid_y_ex = np.reshape(valid_y, (np.shape(valid_y)[0], 1))

		# build dataset
		torch_train = RegressionDataset(x, y_ex)
		train_data_loader = DataLoader(torch_train, batch_size=self.batch_size)
		torch_valid = RegressionDataset(valid_x, valid_y_ex)
		valid_data_loader = DataLoader(torch_valid, batch_size=self.batch_size)

		valid_error_, y_pred_ = \
			self.single_fit_and_predict(train_data_loader, valid_data_loader, test_x)
		test_y = y_pred_
		return test_y

	def fit_and_predict(self, x, y, valid_x, valid_y, test_x):
		test_y = self.model_fit_and_predict(x, y, valid_x, valid_y, test_x)
		test_y = np.reshape(test_y, (np.shape(test_y)[0],))
		return test_y
