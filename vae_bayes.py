import torch
from torch import nn
import torch.distributions as dist
from torch.nn import functional as F
from torch.distributions.kl import kl_divergence


def linear_variational(module, name=None):
    m, with_bias = module, module.bias is not None
    setattr(module, 'name', name)

    mu_W = nn.Parameter(torch.Tensor(m.out_features, m.in_features))
    logvar_W = nn.Parameter(torch.Tensor(m.out_features, m.in_features))

    # register parameter checks if it has been registered, Parameter dos not do that
    m.register_parameter("weight_mu", mu_W)
    m.register_parameter("weight_logvar", logvar_W)

    var = 2 / (m.out_features + m.in_features)
    nn.init.normal_(mu_W, 0.0, std=var ** (1 / 2))
    nn.init.constant_(logvar_W, -5)

    if with_bias:
        mu_bias = nn.Parameter(torch.Tensor(m.out_features))
        logvar_bias = nn.Parameter(torch.Tensor(m.out_features))

        nn.init.constant_(mu_bias, 0.1)
        nn.init.constant_(logvar_bias, -10)

        m.register_parameter('bias_mu', mu_bias)
        m.register_parameter('bias_logvar', logvar_bias)

        del m._parameters['bias']  # <- we are going to recreate it
        setattr(m, 'bias', None)  # when sampling for new weights

    m.register_forward_pre_hook(sample_new_params)
    return m


def sample_new_params(module, input=None):
    m = module

    if ('weight' in m._parameters.keys()):
        del m._parameters['weight']

    sample = dist.Normal(m.weight_mu, m.weight_logvar.mul(1 / 2).exp()).rsample()
    setattr(m, "weight", sample)

    if hasattr(m, 'bias_mu'):
        sample = dist.Normal(m.bias_mu, m.bias_logvar.mul(1 / 2).exp()).rsample()
        setattr(m, 'bias', sample)

    return None


class VAE(torch.nn.Module):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()
        self.variational_layers = []

        self.hidden_size = 2000
        self.latent_size = 30
        self.alphabet_size = kwargs['alphabet_size']
        self.seq_len = kwargs['seq_len']
        self.input_size = self.alphabet_size * self.seq_len
        self.dropout = 0.0
        self.Neff = kwargs['Neff']
        self.div = 4
        self.inner = 40
        self.h2_div = 4
        self.bayesian = True

        # Encoder
        self.fc1 = torch.nn.Linear(self.input_size, int(self.hidden_size * (3 / 4)))
        self.dpe1 = torch.nn.Dropout(self.dropout)

        self.fc1h = torch.nn.Linear(int(self.hidden_size * (3 / 4)), int(self.hidden_size * (3 / 4)))
        self.dpe2 = torch.nn.Dropout(self.dropout)

        # Latent
        self.fc21 = torch.nn.Linear(int(self.hidden_size * (3 / 4)), self.latent_size)
        self.fc22 = torch.nn.Linear(int(self.hidden_size * (3 / 4)), self.latent_size)

        # Decoder
        self.fc3 = torch.nn.Linear(self.latent_size, self.hidden_size // 4)
        # self.fc3 = linear_variational(self.fc3)
        # self.variational_layers.append(self.fc3)
        # self.dpd1 = torch.nn.Dropout(self.dropout)

        self.fc3h = torch.nn.Linear(self.hidden_size // 4, self.hidden_size // self.h2_div)
        # self.fc3h = linear_variational(self.fc3h)
        # self.variational_layers.append(self.fc3h)
        # self.dpd2 = torch.nn.Dropout(self.dropout)

        # Group Sparsity
        self.W = torch.nn.Linear(self.inner, (self.hidden_size // self.h2_div) * self.seq_len, bias=False)  # W
        self.C = torch.nn.Linear(self.alphabet_size, self.inner, bias=False)  # D
        self.S = torch.nn.Linear(self.seq_len, (self.hidden_size // self.h2_div // self.div), bias=False)  # S

        if self.bayesian:
            self.W = linear_variational(self.W, 'w')
            self.C = linear_variational(self.C, 'C')
            self.S = linear_variational(self.S, 'S')
            self.variational_layers.append(self.W)
            self.variational_layers.append(self.C)
            self.variational_layers.append(self.S)

            lamb_W_out_b_dim = self.input_size

            self.lamb_mu = torch.nn.Parameter(torch.Tensor([1] * 1))
            self.lamb_logvar = torch.nn.Parameter(torch.Tensor([-5] * 1))
            self.W_out_b_mu = torch.nn.Parameter(torch.Tensor([0.1] * lamb_W_out_b_dim))
            self.W_out_b_logvar = torch.nn.Parameter(torch.Tensor([-5] * lamb_W_out_b_dim))
        else:
            self.lamb = torch.nn.Parameter(torch.Tensor([0.1] * self.input_size))
            self.W_out_b = torch.nn.Parameter(torch.Tensor([0.1]))

        for layer in self.children():
            if not str(layer).startswith('Linear'):
                continue

            torch.nn.init.xavier_normal_(layer.weight)

            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.constant_(self.fc22.bias, -5)

    def logp(self, batch, rand=False):
        x = batch.view(-1, self.input_size)
        mu, logvar = self.encoder(x)
        z = dist.Normal(mu, logvar.mul(1 / 2).exp()).rsample() if rand else mu

        recon = self.decoder(z)
        recon = recon.view(-1, self.alphabet_size, self.seq_len)
        recon = recon.log_softmax(dim=1)

        logp = (-recon * batch).sum(-2).sum(-1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)
        elbo = logp + kl

        return elbo

    def loss(self, x_hat, true_x, mu, logvar, beta=1):
        RL = (-x_hat * true_x).sum(-2).sum(-1)  # reconst. loss

        # loss for z - latent
        p_q = dist.Normal(mu, logvar.mul(1 / 2).exp())
        p_p = dist.Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        KLZ = kl_divergence(p_q, p_p).sum(-1)

        # loss for network parameters
        KLP = 0
        for l in self.variational_layers:
            w_mu, w_std = l.weight_mu, l.weight_logvar.mul(1 / 2).exp()
            q_w = dist.Normal(w_mu, w_std)

            if l.name == 'S':
                p_w = dist.Normal(torch.zeros_like(w_mu) - 9.305, torch.zeros_like(w_std) + 4)
            else:
                p_w = dist.Normal(torch.zeros_like(w_mu), torch.ones_like(w_std))

            KLP += kl_divergence(q_w, p_w).sum()

            if l.bias is None: continue

            b_mu, b_std = l.bias_mu, l.bias_logvar.mul(1 / 2).exp()
            q_b = dist.Normal(b_mu, b_std)
            p_b = dist.Normal(torch.zeros_like(b_mu), torch.ones_like(b_std))

            KLP += kl_divergence(q_b, p_b).sum()

        lamb_mu, lamb_std = self.lamb_mu, self.lamb_logvar.mul(1 / 2).exp()
        q_lamb = dist.Normal(lamb_mu, lamb_std)
        p_lamb = dist.Normal(torch.zeros_like(lamb_mu), torch.ones_like(lamb_std))
        KLP += kl_divergence(q_lamb, p_lamb).sum()

        W_out_b_mu, W_out_b_std = self.W_out_b_mu, self.W_out_b_logvar.mul(1 / 2).exp()
        q_W_out_b = dist.Normal(W_out_b_mu, W_out_b_std)
        p_W_out_b = dist.Normal(torch.zeros_like(W_out_b_mu), torch.ones_like(W_out_b_std))
        KLP += kl_divergence(q_W_out_b, p_W_out_b).sum()

        KLP /= self.Neff

        loss = (RL + beta * KLZ).mean() + KLP

        KLP = torch.tensor([0], requires_grad=False) if KLP == 0 else KLP

        return loss, RL.mean(), KLZ.mean(), KLP

    def encoder(self, x):
        x = x.view(-1, self.input_size)  # flatten
        h = F.relu(self.fc1(x))  # input layer
        h = self.dpe1(h)
        h = F.relu(self.fc1h(h))  # encoder hidden layer
        h = self.dpe2(h)

        return self.fc21(h), self.fc22(h)

    def decoder(self, x):
        h = self.fc3(x)
        h = F.relu(h)  # First hidden layer after bottleneck
        # h = self.dpd1(h)  # Dropout
        h = self.fc3h(h)
        h = torch.sigmoid(h)  # Second hidden layer after bottleneck
        # h = self.dpd2(h)

        if self.bayesian:
            for module in [self.W, self.C, self.S]:
                sample_new_params(module)

            lamb = dist.Normal(self.lamb_mu, self.lamb_logvar.mul(1 / 2).exp()).rsample()
            b = dist.Normal(self.W_out_b_mu, self.W_out_b_logvar.mul(1 / 2).exp()).rsample()
        else:
            lamb = self.lamb
            b = self.W_out_b

        W = self.W.weight
        C = self.C.weight
        S = self.S.weight.repeat(self.div, 1)
        S = torch.sigmoid(S)

        W_out = torch.mm(W, C)
        order = (self.hidden_size // self.h2_div, self.alphabet_size, self.seq_len)
        W_out = W_out.view(*order)
        S = S.unsqueeze(-2)  # S :      [hidden x 1 x seq_len()]
        W_out = W_out * S  # W_out :  [hidden x alphabet_size x sew_len]
        W_out = W_out.view(-1, self.input_size)  # W_out :  [hidden x input_size]

        # parameterize each final weight matrix as W^(3,i) /paper
        return (1 + lamb.exp()).log() * F.linear(h, W_out.T, b)

    def forward(self, x, rep=True):
        mu, logvar = self.encoder(x)  # encode

        if rep:
            x = dist.Normal(mu, logvar.mul(1 / 2).exp()).rsample()  # reparameterize
        else:  # or don't
            x = mu

        x = self.decoder(x)  # decode
        x = x.view(-1, self.alphabet_size, self.seq_len)  # squeeze back
        x = x.log_softmax(dim=1)  # softmax

        return x, mu, logvar