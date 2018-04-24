from __future__ import print_function
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

# Params
LEAK = 0.05

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class Unflatten(nn.Module):
    def __init__(self, n=1):
        super(Unflatten, self).__init__()
        self.n = n

    def forward(self, x):
        return x.view(x.size()[0], self.n, -1)

# VAE model
class VAE(nn.Module):
    def __init__(self, freq_size=1025, filters=8, z_dim=256, enable_cuda=False):
        super(VAE, self).__init__()
        self.freq_size = freq_size
        self.filters = filters
        self.z_dim = z_dim
        self.enable_cuda = enable_cuda
        inter = freq_size / 16 * filters
        self.encoder = nn.Sequential(OrderedDict([ # if in (n, 1025, 2)
            ('conv1', VAE.conv(2, filters, 5, 2, 2)), # out (n, 513, 8)
            ('conv2', VAE.conv(filters, filters*2, 5, 2, 2)), # out (n, 257, 16)
            ('conv3', VAE.conv(filters*2, filters*4, 5, 2, 2)), # out (n, 129, 32)
            ('conv4', VAE.conv(filters*4, filters*4, 5, 2, 2)), # out (n, 65, 32)
            ('conv5', VAE.conv(filters*4, filters*4, 5, 2, 2)), # out (n, 33, 32)
            ('conv6', VAE.conv(filters*4, filters*4, 5, 2)), # out (n, 16, 32)
            ('flat', Flatten()), # out (n, 512)
            ('fc1', nn.Linear(inter, inter)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(inter, 2 * z_dim)) # 2 for mean and variance.
        ]))

        self.decoder = nn.Sequential(OrderedDict([ # if in (n, z_dim)
            ('fc', nn.Linear(z_dim, inter)), # out (n, 512)
            ('relu', nn.ReLU()),
            ('unflat', Unflatten(filters*4)), # out (n, 16, 32)
            ('deconv1', VAE.deconv(filters*4, filters*4, 5, 2)), # out (n, 33, 32)
            ('deconv2', VAE.deconv(filters*4, filters*4, 5, 2, 2)), # out (n, 65, 32)
            ('deconv3', VAE.deconv(filters*4, filters*4, 5, 2, 2)), # out (n, 129, 32)
            ('deconv4', VAE.deconv(filters*4, filters*2, 5, 2, 2)), # out (n, 257, 16)
            ('deconv5', VAE.deconv(filters*2, filters, 5, 2, 2)), # out (n, 513, 8)
            ('deconv6', VAE.deconv(filters, 2, 5, 2, 2, bn=False)), # out (n, 1025, 2)
        ]))
        if enable_cuda:
            self.cuda()

    @staticmethod
    def load(filename, enable_cuda=False):
        state = torch.load(filename)
        vae = VAE(state['freq_size'], state['filters'], state['z_dim'], enable_cuda)
        vae.vae_load_state_dict(state)
        return vae

    def save(self, filename):
        torch.save(self.vae_state_dict(), filename)

    def vae_load_state_dict(self, state_dict):
        self.freq_size = state_dict['freq_size']
        self.filters = state_dict['filters']
        self.z_dim = state_dict['z_dim']
        self.load_state_dict(state_dict['model'])

    def vae_state_dict(self):
        return {
            'freq_size':self.freq_size,
            'filters':self.filters,
            'z_dim':self.z_dim,
            'model':self.state_dict()
        }

    def to_var(self, x, **kw):
        if self.enable_cuda:
            x = x.cuda()
        return Variable(x, **kw)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)  # mean and log variance.
        z = self.reparameterize(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var

    def sample(self, z):
        return self.decoder(z)

    @staticmethod
    def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
        layers = []
        layers.append(nn.Conv1d(c_in, c_out, k_size, stride, pad))
        if bn:
            layers.append(nn.BatchNorm1d(c_out))
        layers.append(nn.LeakyReLU(LEAK))
        return nn.Sequential(*layers)

    @staticmethod
    def deconv(c_in, c_out, k_size, stride=2, pad=1, o=0, bn=True):
        layers = []
        layers.append(nn.ConvTranspose1d(c_in, c_out, k_size, stride, pad, output_padding=o))
        if bn:
            layers.append(nn.BatchNorm1d(c_out))
        layers.append(nn.LeakyReLU(LEAK))
        return nn.Sequential(*layers)

def train_vae(vae, train_loader, valid_loader, epochs, lr, results_cb=None):
    optimizer = torch.optim.Adam(vae.parameters(), lr)

    def adjust_learning_rate(epoch):
        lr_ = lr * (0.1 ** (epoch / 100.0))
        print(lr_)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

    def format_result_dict(r):
        return ("Epoch [{epoch: 3}/{epochs}], "
                "Training Losses "
                "[total={train_total_loss}, "
                "reconst={train_reconst_loss}, "
                "kl={train_kl_loss}], "
                "Validation Losses "
                "[total={valid_total_loss}, "
                "reconst={valid_reconst_loss}, "
                "kl={valid_kl_loss}]"
        ).format(**r)

    print("Beginning to train the VAE")
    for epoch in range(epochs):
        adjust_learning_rate(epoch)

        train_reconst_loss, train_kl_loss = run_vae(vae, train_loader, optimizer)
        valid_reconst_loss, valid_kl_loss = run_vae(vae, valid_loader)

        if results_cb is not None:
            result_dict = {
                'epoch':epoch+1,
                'epochs':epochs,
                'train_total_loss':train_reconst_loss + train_kl_loss,
                'train_reconst_loss':train_reconst_loss,
                'train_kl_loss':train_kl_loss,
                'valid_total_loss':valid_reconst_loss + valid_kl_loss,
                'valid_reconst_loss':valid_reconst_loss,
                'valid_kl_loss':valid_kl_loss
            }
            results_cb(result_dict)
        print(format_result_dict(result_dict))

def run_vae(vae, loader, optimizer=None, keep_results=False):
    train = optimizer is not None
    dataset_size = sum(len(batch) for batch in loader)
    reconst_loss_mean = Mean(dataset_size)
    kl_loss_mean = Mean(dataset_size)
    vae.train(train)
    batch_results = []

    for i, stfted in enumerate(loader):
        stfted = vae.to_var(stfted, volatile=not train)
        out, mu, log_var = vae(stfted)
        batch_size = stfted.data.shape[0]

        # Compute reconstruction loss and kl divergence
        reconst_loss = torch.sum(torch.mean((stfted - out)**2, 0))
        kl_loss = 0.5 * torch.sum(torch.mean(torch.exp(log_var) + mu**2 - 1. - log_var, 0))
        total_loss = reconst_loss + kl_loss

        if train:
            # Backprop + Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        elif keep_results:
            batch_results.append(out.cpu())

        reconst_loss_mean.append(reconst_loss.data[0], batch_size)
        kl_loss_mean.append(kl_loss.data[0], batch_size)

    if not keep_results:
        return reconst_loss_mean.acc, kl_loss_mean.acc
    else:
        return torch.cat(batch_results), reconst_loss_mean.acc, kl_loss_mean.acc

class Mean:
    def __init__(self, total=None):
        self.acc = 0
        self.total = float(total)

    def append(self, num, denom):
        self.acc += num * (float(denom) / self.total)
