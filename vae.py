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
        self.enable_cuda = enable_cuda
        if enable_cuda:
            self.cuda()

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

def train_vae(vae, data_loader, epochs, lr, results_cb=None):
    vae.train()
    optimizer = torch.optim.Adam(vae.parameters(), lr)
    iter_per_epoch = len(data_loader)
    def adjust_learning_rate(epoch):
        lr_ = lr * (0.1 ** (epoch / 50.0))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
            
    for epoch in range(epochs):
        adjust_learning_rate(epoch)
        for i, stfted in enumerate(data_loader):

            stfted = vae.to_var(stfted)
            out, mu, log_var = vae(stfted)
            batch_size = stfted.data.shape[0]

            # Compute reconstruction loss and kl divergence
            reconst_loss = torch.sum(torch.mean((stfted - out)**2, 0))
            kl_loss = 0.5 * torch.sum(torch.mean(torch.exp(log_var) + mu**2 - 1. - log_var, 0))
        
            # Backprop + Optimize
            total_loss = reconst_loss + kl_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if results_cb is not None:
                result_dict = {
                    'epoch':epoch+1,
                    'epochs':epochs,
                    'iter':i+1,
                    'iters':iter_per_epoch,
                    'total_loss':total_loss.data[0],
                    'reconst_loss':reconst_loss.data[0],
                    'kl_loss':kl_loss.data[0]
                }
                results_cb(result_dict)
            
            if i % 100 == 0:
                print ("Epoch[%.2d/%d]," % (epoch+1, epochs),
                       "Step [%.4d/%d]," % (i+1, iter_per_epoch),
                       "Total Loss: %.4f," % (total_loss.data[0]),
                       "Reconst Loss: %.4f," % (reconst_loss.data[0]),
                       "KL Div: %.7f" % (kl_loss.data[0]))

def eval_result(vae, dataloader):
    batch_results = []
    for i, stfted in enumerate(dataloader):
        stfted = vae.to_var(stfted, volatile=True)
        out, mu, log_var = vae(stfted)

        batch_results.append(out.cpu())
        del out, mu, log_var, stfted

    return torch.cat(batch_results)
