import torch
import torch.distributions as dist


class LinearVariational(torch.nn.Module):
    """
    Mean field approximation of nn.Linear
    """

    def __init__(self, in_features, out_features, parent, bias=True, layer_s=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.include_bias = bias
        self.parent = parent
        self.layer_s = layer_s

        ##### priors initialization
        # set the accumulated kl of the parent kl_loss to 0
        if getattr(parent, 'accumulated_kl_div', None) is None:
            parent.accumulated_kl_div = 0


        # init variance
        variance = 2 / (out_features + in_features)

        # init weight mu prior
        self.w_mu = torch.nn.Parameter(
            torch.Tensor(out_features, in_features).normal_(mean=0.0, std=variance ** (1 / 2))
        )

        # init weight variance prior
        self.w_p = torch.nn.Parameter(
            (-5) * torch.ones([out_features, in_features])
        )

        # check if bias is included
        if self.include_bias:
            # init bias mu prior
            self.b_mu = torch.nn.Parameter(0.1 * torch.ones(out_features))

            # init bias variance prior
            self.b_p = torch.nn.Parameter((-10) * torch.ones(out_features))


    def kl_divergence(self, p_q, mu_theta, p_theta, layer_s):
        if layer_s:
            p_p = dist.Normal(torch.zeros_like(mu_theta) - 9.305, torch.zeros_like(p_theta) + 4)
        else:
            p_p = dist.Normal(torch.zeros_like(mu_theta), torch.ones_like(p_theta))
        return torch.distributions.kl.kl_divergence(p_p, p_q).sum()

    def forward(self, x):
        q_w = dist.Normal(self.w_mu, self.w_p.mul(1 / 2).exp())
        w = q_w.rsample()

        self.parent.accumulated_kl_div = 0

        # KLP weight
        self.parent.accumulated_kl_div += self.kl_divergence(q_w, self.w_mu, self.w_p, self.layer_s)

        if self.include_bias:
            q_b = dist.Normal(self.b_mu, self.b_p.mul(1 / 2).exp())
            b = q_b.rsample()

            # KLP bias
            self.parent.accumulated_kl_div += self.kl_divergence(q_b, self.b_mu, self.b_p, self.layer_s)
        else:
            b = 0

        z = x @ torch.transpose(w, 0, 1) + b

        return z
