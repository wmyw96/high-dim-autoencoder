import torch
from torch import nn
from collections import OrderedDict


class AutoEncoder(nn.Module):
	def __init__(self, dim_x, dim_z, encoder_hidden, decoder_hidden, latent_act):
		super(AutoEncoder, self).__init__()

		self.dim_x = dim_x
		self.dim_z = dim_z
		self.encoder_hidden = encoder_hidden
		self.decoder_hidden = decoder_hidden

		def build_nn_module(name, cfg):
			relu_nn = []
			depth = len(cfg) - 2
			for i in range(depth):
				relu_nn.append(('{}linear{}'.format(name, i), nn.Linear(cfg[i], cfg[i + 1])))
				relu_nn.append(('{}relu{}'.format(name, i), nn.ReLU()))
			relu_nn.append(('{}linear{}'.format(name, depth), nn.Linear(cfg[depth], cfg[depth + 1])))
			return relu_nn

		encoder_nn = build_nn_module('encoder', [dim_x] + encoder_hidden + [dim_z])
		decoder_nn = build_nn_module('decoder', [dim_z] + decoder_hidden + [dim_x])
		self.encoder = nn.Sequential(OrderedDict(encoder_nn))
		self.decoder = nn.Sequential(OrderedDict(decoder_nn))

		if latent_act is None:
			self.latent_act_func = torch.nn.Identity()
		elif latent_act == 'relu':
			self.latent_act_func = torch.nn.ReLU()
		elif latent_act == 'tanh':
			self.latent_act_func = torch.nn.Tanh()
		else:
			raise ValueError("wrong argument latent_act {}".format(latent_act))

	def forward(self, x, is_training=False, out_z=False):
		pre_act_h = self.encoder(x)
		acted_h = self.latent_act_func(pre_act_h)
		x_hat = self.decoder(acted_h)
		if out_z:
			return pre_act_h
		else:
			return x_hat

