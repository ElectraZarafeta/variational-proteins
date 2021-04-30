import torch
import torch.distributions as dist
from linear_variational import LinearVariational
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F


class KL:
    accumulated_kl_div = 0


class VAE(torch.nn.Module):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()
        self.hidden_size = 32
        self.latent_size = 2
        self.alphabet_size = kwargs['alphabet_size']
        self.seq_len = kwargs['seq_len']
        self.input_size = self.alphabet_size * self.seq_len
        self.kl_loss = KL  # set accumulated kl to 0
        self.dropout = 0.0
        self.Neff = kwargs['Neff']
        self.div = 8
        self.inner = 16
        self.h2_div = 1
        self.bayesian = True

        self.encoder_seq = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, int(self.hidden_size * (3 / 4))),
            torch.nn.Dropout(self.dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(int(self.hidden_size * (3 / 4)), int(self.hidden_size * (3 / 4))),
            torch.nn.Dropout(self.dropout),
            torch.nn.ReLU(),
        )

        self.fc21 = torch.nn.Linear(int(self.hidden_size * (3 / 4)), self.latent_size)
        self.fc22 = torch.nn.Linear(int(self.hidden_size * (3 / 4)), self.latent_size)

        self.decoder_seq = torch.nn.Sequential(
            LinearVariational(self.latent_size, self.hidden_size // 16, self.kl_loss),
            torch.nn.Dropout(self.dropout),
            torch.nn.ReLU(),
            LinearVariational(self.hidden_size // 16, self.hidden_size // self.h2_div, self.kl_loss),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(self.dropout),
        )

        # Group Sparsity
        if self.bayesian:
            self.W = LinearVariational(self.inner, (self.hidden_size // self.h2_div) * self.seq_len, self.kl_loss)  # W
            self.C = LinearVariational(self.alphabet_size, self.inner, self.kl_loss)  # D
            self.S = LinearVariational(self.seq_len, (self.hidden_size // self.h2_div // self.div), self.kl_loss,
                                       layer_s=True)  # S

            lamb_W_out_b_dim = self.input_size

            self.lamb_mu = torch.nn.Parameter(torch.Tensor([1] * 1))
            self.lamb_logvar = torch.nn.Parameter(torch.Tensor([-5] * 1))
            self.W_out_b_mu = torch.nn.Parameter(torch.Tensor([0.1] * lamb_W_out_b_dim))
            self.W_out_b_logvar = torch.nn.Parameter(torch.Tensor([-5] * lamb_W_out_b_dim))
        else:
            self.lamb = torch.nn.Parameter(torch.Tensor([0.1] * self.input_size))
            self.W_out_b = torch.nn.Parameter(torch.Tensor([0.1]))

        for layer in self.children():
            if str(layer) == 'Linear':
                torch.nn.init.xavier_normal_(layer.weight)

                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.constant_(self.fc22.bias, -5)

    def accumulated_kl_div(self):
        return self.kl_loss.accumulated_kl_div

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
        RL = -(x_hat * true_x).sum(-2).sum(-1)  # reconst. loss

        # loss for z - latent
        p_q = dist.Normal(mu, logvar.mul(1 / 2).exp())
        p_p = dist.Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        KLZ = kl_divergence(p_q, p_p).sum(-1)

        # loss for network parameters
        KLP = 0

        lamb_mu, lamb_std = self.lamb_mu, self.lamb_logvar.mul(1 / 2).exp()
        q_lamb = dist.Normal(lamb_mu, lamb_std)
        p_lamb = dist.Normal(torch.zeros_like(lamb_mu), torch.ones_like(lamb_std))
        KLP += kl_divergence(q_lamb, p_lamb).sum()

        W_out_b_mu, W_out_b_std = self.W_out_b_mu, self.W_out_b_logvar.mul(1 / 2).exp()
        q_W_out_b = dist.Normal(W_out_b_mu, W_out_b_std)
        p_W_out_b = dist.Normal(torch.zeros_like(W_out_b_mu), torch.ones_like(W_out_b_std))
        KLP += kl_divergence(q_W_out_b, p_W_out_b).sum()

        KLP += self.kl_loss.accumulated_kl_div
        KLP /= self.Neff

        loss = (RL + beta * KLZ).mean() + KLP

        KLP = torch.tensor([0], requires_grad=False) if KLP == 0 else KLP

        return loss, RL.mean(), KLZ.mean(), KLP

    def encoder(self, x):
        x = self.encoder_seq(x)
        mu, logvar = self.fc21(x), self.fc22(x)  # branch mu, var

        return mu, logvar

    def decoder(self, x):
        x = self.decoder_seq(x)
        include_bias = True

        if self.bayesian:
            for module in [self.W, self.C, self.S]:
                m = module

                q_w = dist.Normal(m.w_mu, m.w_p.mul(1 / 2).exp())
                setattr(m, "weight", q_w.rsample())

                if include_bias:
                    q_b = dist.Normal(m.b_mu, m.b_p.mul(1 / 2).exp())
                    setattr(m, "bias", q_b.rsample())
                else:
                    setattr(m, "bias", None)

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

        return (1 + lamb.exp()).log() * F.linear(x, W_out.T,
                                                 b)  # parameterize each final weight matrix as W^(3,i) /paper

    def forward(self, x, rep=True):
        x = x.view(-1, self.input_size)  # flatten
        mu, logvar = self.encoder(x)  # encode

        if rep:
            x = dist.Normal(mu, logvar.mul(1 / 2).exp()).rsample()  # reparameterize
        else:  # or don't
            x = mu

        x = self.decoder(x)  # decode
        x = x.view(-1, self.alphabet_size, self.seq_len)  # squeeze back
        x = x.log_softmax(dim=1)  # softmax

        return x, mu, logvar
