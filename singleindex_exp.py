from colorama import init, Fore
import torch
import random
import numpy as np
from torch import nn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from data.generative_model import NonlinearFactorModelSingleIndex, GenerativeModelDataset
from estimators.regression_nn import RegressionNNEstimator
from torch.utils.data import DataLoader
from model import AutoEncoder
from utils import *

import argparse
import time

init(autoreset=True)
parser = argparse.ArgumentParser()
parser.add_argument("--n", help="number of samples", type=int, default=1000)
parser.add_argument("--p", help="data dimension", type=int, default=1000)
parser.add_argument("--r", help="factor dimension", type=int, default=1)
parser.add_argument('--rz', help='estimated factor dimension', type=int, default=1)
parser.add_argument("--batch_size", help="batch size", type=int, default=64)

parser.add_argument("--seed", help="random seed", type=int, default=0)
args = parser.parse_args()

encoder_hiddens = []
decoder_hiddens = []

start_time = time.time()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

n_train = args.n
n_valid = n_train * 3 // 7
p = args.p
r = args.r
R = args.rz
batch_size = args.batch_size
n_test = 10000


def ident(x):
	return x


nonlinear_factor_model = NonlinearFactorModelSingleIndex(p, r, index_func=np.sin)


def nonlinear_factor_data(n):
	z = np.reshape(np.random.uniform(-1, 1, n * r), (n, r))
	x = nonlinear_factor_model.sample(z)
	return x, z


x_train, _ = nonlinear_factor_data(n_train)
x_test, _ = nonlinear_factor_data(n_test)

x_train_reg, f_train_reg = nonlinear_factor_data(n_train)
x_valid_reg, f_valid_reg = nonlinear_factor_data(n_valid)
x_test_reg, f_test_reg = nonlinear_factor_data(n_test)
y_train_reg = np.sin(np.sum(f_train_reg, 1)) + np.random.normal(0, 1, n_train)
y_valid_reg = np.sin(np.sum(f_valid_reg, 1)) + np.random.normal(0, 1, n_valid)
y_test_reg = np.sin(np.sum(f_test_reg, 1))

train_data = GenerativeModelDataset(x_train)
test_data = GenerativeModelDataset(x_test)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
ae_model = AutoEncoder(dim_x=p, dim_z=R, encoder_hidden=encoder_hiddens,
						decoder_hidden=decoder_hiddens, latent_act='relu').to(device)
print(f'Autoencoder Model:\n {ae_model}')
learning_rate = 1e-3
num_epoch = 300


def train_loop(data_loader, model, loss_fn, optimizer):
	loss_rec = {'l2_loss': 0.0}
	for batch, x in enumerate(data_loader):
		x_hat = model(x, is_training=True)
		loss = loss_fn(x_hat, x)
		loss_rec['l2_loss'] += loss.item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	loss_rec['l2_loss'] /= len(data_loader)
	return loss_rec


def test_loop(data_loader, model, loss_fn):
	loss_sum = 0
	with torch.no_grad():
		for x in data_loader:
			x_hat = model(x, is_training=False)
			loss_sum += loss_fn(x_hat, x).item()
	loss_rec = {'l2_loss': loss_sum / len(data_loader)}
	return loss_rec


ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=learning_rate)
ae_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=ae_optimizer, gamma=0.995)
mse_loss = nn.MSELoss()

# orcale regression error
nn_est_oracle = RegressionNNEstimator(dim_x=r, depth=3, width=100)
y_oracle = nn_est_oracle.fit_and_predict(f_train_reg, y_train_reg, f_valid_reg, y_valid_reg, f_test_reg)
mse_oracle = np.mean(np.square(y_oracle - y_test_reg))
print(f'oracle regression task error = {mse_oracle}')

for epoch in range(num_epoch):
	if epoch % 10 == 0:
		test_losses = test_loop(test_dataloader, ae_model, mse_loss)
		print(f"[Test]   " + unpack_loss(test_losses))
		# test via regression task
		with torch.no_grad():
			tilde_f_train_reg = ae_model(torch.tensor(x_train_reg, dtype=torch.float32), out_z=True).detach().numpy()
			tilde_f_valid_reg = ae_model(torch.tensor(x_valid_reg, dtype=torch.float32), out_z=True).detach().numpy()
			tilde_f_test_reg = ae_model(torch.tensor(x_test_reg, dtype=torch.float32), out_z=True).detach().numpy()
		print(np.shape(tilde_f_valid_reg))
		nn_est1 = RegressionNNEstimator(dim_x=R, depth=3, width=100)
		y1 = nn_est1.fit_and_predict(tilde_f_train_reg, y_train_reg, tilde_f_valid_reg, y_valid_reg, tilde_f_test_reg)
		mse = np.mean(np.square(y1 - y_test_reg))
		print(f'regression task error = {mse}')
	train_losses = train_loop(train_dataloader, ae_model, mse_loss, ae_optimizer)
	print(f"[Train]    " + unpack_loss(train_losses))
