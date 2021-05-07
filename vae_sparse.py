# This VAE is as vanilla as it can be.
import torch, sys, os, math
import numpy as np
import pandas as pd 
import sw.forwardpass as fp 
from collections import defaultdict
from torch import nn, optim 
from torch.nn import functional as F 

from torch.distributions import kl_divergence
from torch.distributions.normal import Normal

class VAE(torch.nn.Module):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()
        self.variational_layers = [] #new line

        self.hidden_size   = 64
        self.latent_size   = 2
        self.alphabet_size = kwargs['alphabet_size']
        self.seq_len       = kwargs['seq_len']
        self.neff          = kwargs['neff']
        self.input_size    = self.alphabet_size * self.seq_len

        # Extra initialization 
        self.div           = dict.get(kwargs, 'div', 8)
        self.beta          = dict.get(kwargs, 'beta', 1)
        self.dropout       = dict.get(kwargs, 'dropout', 0.0) #dropout at zero
        self.latent_size   = dict.get(kwargs, 'latent_size', 32)
        self.inner         = dict.get(kwargs, 'inner', 16)
        self.h2_div        = dict.get(kwargs, 'h2_div', 1)
        self.bayesian      = dict.get(kwargs, 'bayesian', True)

        # self.encoder = torch.nn.Sequential(
        #     torch.nn.Linear(self.input_size, self.hidden_size),
        #     torch.nn.ReLU(),
        # )
        
        # Encoder 
        self.fc1           = nn.Linear(self.input_size, int(self.hidden_size * (3/4))) #why mul 0.75?
        self.dpe1          = nn.Dropout(self.dropout)

        self.fc1h          = nn.Linear(int(self.hidden_size * (3/4)), int(self.hidden_size * (3/4)))
        self.fpe2          = nn.Dropout(self.dropout)


        # Latent space `mu` and `var`
        self.fc21 = nn.Linear(int(self.hidden_size * (3/4)), self.latent_size)
        self.fc22 = nn.Linear(int(self.hidden_size * (3/4)), self.latent_size)

        # self.decoder = torch.nn.Sequential(
        #     torch.nn.Linear(self.latent_size, self.hidden_size),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.hidden_size, self.input_size),
        # )

        # Decoder 
         
        self.fc3            = nn.Linear(self.latent_size, self.hidden_size // 16)
        # edw mpainei to make_variational_linear, 
        # kai to kanei append sto variational layers list 
        # self.fc3 = make_variational_linear(self.fc3)
        # self.variational_layers.append(self.fc3)
        self.dpd1 = nn.Dropout(self.dropout)

        self.fc3h = nn.Linear(self.hidden_size // 16, self.hidden_size // self.h2_div)
        # edw mpainei to make_variational_linear, 
        # kai to kanei append sto variational layers list 
        # self.fc3 = make_variational_linear(self.fc3)
        # self.variational_layers.append(self.fc3)
        self.dpd2 = nn.Dropout(self.dropout)


        # Group Sparsity
        self.W = nn.Linear(self.inner, (self.hidden_size // self.h2_div) * self.seq_len, bias=False) #W
        self.C = nn.Linear(self.alphabet_size, self.inner, bias=False) # D
        self.S = nn.Linear(self.seq_len, (self.hidden_size // self.h2_div // self.div, bias=False)) # S

        if self.bayesian: 
            self.W = make_variational_linear(self.W, 'w')
            self.C = make_variational_linear(self.C, 'C')
            self.S = make_variational_linear(self.S, 's')
            self.variational_layers.append(self.W)
            self.variational_layers.append(self.C)
            self.variational_layers.append(self.S)

            lamb_W_out_b_dim = self.input_size

            self.lamb_mean = nn.Parameter(torch.Tensor([1], 1))
            self.lamb_logvar = nn.Parameter(torch.Tensor([-5] * 1))
            self.W_out_b_mean = nn.Parameter(torch.Tensor([0.1] * lamb_W_out_b_dim))
            self.W_out_b_logvar = nn.Parameter(torch.Tensor([-5] * lamb_W_out_b_dim))
        else:
            self.lamb = nn.Parameter(torch.Tensor([0.1] * self.input_size))
            self.W_out_b = nn.Parameter(torch.Tensor([0.1]))

        # INIT 
        for layer in self.children():
            if not str(layer).startswith('Linear'):
                continue

            nn.init.xavier_normal_(layer.weight)

            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.1)

        nn.init.constant_(self.fc22.bias, -5)

    def encode(self, x):
        x = x.view(-1, self.input_size) #batch x flat_one_hot
        h = F.relu(self, fc1(x)) #input layer
        h = self.dpe1(h)
        h = F.relu(self.fc1h(h)) #encoder hidden layer
        h = self.dpe2(h)

        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        return Normal(mu, logvar.mul(1/2).exp()).rsample()

    def decode(self, z):
        h = self.fc3(z)
        h = F.relu(h)       #First hidden layer after bottleneck
        h = self.dpd1(h)    #Dropout
        h = torch.sigmoid(self.fc3h(h)) #Second hidden layer after bottleneck
        h = self.dpd2(h)    #Dropout

        if self.bayesian:
            for module in [self.W, self.C, self.S]:     # we need to sample the weight manually
                sample_new_weight(module, None)         # because we never call forward on those layers

            lamb = Normal(self.lamb_mean, self.lamb_logvar.mul(1/2).exp()).rsample()
            b = Normal(self.W_out_b_mean, self.W_out_b_logvar.mul(1/2).exp()).rsample()
        else: 
            lamb = self.lamb
            b = self.W_out_b

        W = self.W.weight
        C = self.C.weight
        S = self.S.weight.repeat(self.div, 1)
        S = torch.sigmoid(S)

        W_out = torch.mm(W, c)
        order = (self.hidden_size // self.h2_div, self.alphabet_size, self.seq_len)
        W_out = W_out.view(*order)
        S     = S.unsqueeze(-2)                  # S :      [hidden x 1 x seq_len()]
        W_out = W_out * S                        # W_out :  [hidden x alphabet_size x sew_len]
        W_out = W_out.view(-1, self.input_size)  # W_out :  [hidden x input_size]           

        return (1 + lamb.exp()).log() * F.linear(h, W_out.T, b) # parameterize each final weight matrix as W^(3,i) /paper

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        out        = self.decode(z)
        out        = out.view(-1, self.alphabet_size, self.seq_len)
        out        = out.log_softmax(dim=1)

        return out, mu, logvar

    # def forward(self, x, rep=True):
    #     x = x.view(-1, self.input_size)                    # flatten
    #     x = self.encoder(x)                                # encode
    #     mu, logvar = self.fc21(x), self.fc22(x)            # branch mu, var

    #     if rep:                                            # reparameterize
    #         x = mu + torch.randn_like(mu) * (0.5*logvar).exp() 
    #     else:                                              # or don't 
    #         x = mu                                         

    #     x = self.decoder(x)                                # decode
    #     x = x.view(-1, self.alphabet_size, self.seq_len)   # squeeze back
    #     x = x.log_softmax(dim=1)                           # softmax
    #     return x, mu, logvar
    
    def loss(self, x_hat, true_x, mu, logvar, beta=0.5):
        RL = -(x_hat*true_x).sum(-1).sum(-1)                    # reconst. loss
        KL = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(-1) # KL loss
        return RL + beta*KL, RL, KL