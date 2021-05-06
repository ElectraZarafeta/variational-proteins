import torch
from misc import data, c
#from vae_bayesian import VAE
from vae_bayes import VAE
from torch import optim
from scipy.stats import spearmanr
import numpy as np
from termcolor import colored


def get_corr_ensample(batch, mutants_values, model, ensamples=128, rand=True):
    model.eval()

    mt_elbos, wt_elbos = 0, 0

    for i in range(ensamples):
        if i and (i % 2 == 0):
            print("\r", f"\tReached {i}/rand={rand}", " " * 32, end='')

        elbos = model.logp(batch, rand=rand).detach()
        wt_elbos += elbos[0]
        mt_elbos += elbos[1:]

    print()

    diffs = (mt_elbos / ensamples) - (wt_elbos / ensamples)
    cor, _ = spearmanr(mutants_values, diffs)

    return cor


batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader, df, mutants_tensor, mutants_df, Neff = data(batch_size=batch_size)

wildtype = dataloader.dataset[0]  # one-hot-encoded wildtype
eval_batch = torch.cat([wildtype.unsqueeze(0), mutants_tensor])

args = {
    'alphabet_size': dataloader.dataset[0].shape[0],
    'seq_len':       dataloader.dataset[0].shape[1],
    'Neff':          Neff,
    'batch_size':    batch_size
}

data = {'dataloader': dataloader, 'df': df}

vae = VAE(**args)
opt = optim.Adam(vae.parameters(), lr=1e-3)

# rl  = Reconstruction loss
# kl  = Kullback-Leibler divergence loss
# cor = Spearman correlation to experimentally measured
#       protein fitness according to eq.1 from paper
stats = {'rl': [], 'klz': [], 'klp': [], 'loss': [], 'cor': []}

for epoch in range(250):
    # Unsupervised training on the MSA sequences.
    vae.train()

    epoch_losses = {'rl': [], 'klz': [], 'klp': [], 'loss': []}
    for batch in dataloader:
        opt.zero_grad()
        x_hat, mu, logvar = vae(batch)
        loss, rl, klz, klp = vae.loss(x_hat, batch, mu, logvar)
        loss.backward()
        opt.step()
        epoch_losses['rl'].append(rl.item())
        epoch_losses['klz'].append(klz.item())
        epoch_losses['klp'].append(klp.item())
        epoch_losses['loss'].append(loss.item())


    if epoch % 16 == 0:
        # Evaluation on mutants
        cor = get_corr_ensample(eval_batch, mutants_df.value, vae)


    # Populate statistics
    stats['rl'].append(np.mean(epoch_losses['rl']))
    stats['klz'].append(np.mean(epoch_losses['klz']))
    stats['klp'].append(np.mean(epoch_losses['klp']))
    stats['loss'].append(np.mean(epoch_losses['loss']))
    stats['cor'].append(np.abs(cor))

    to_print = [
        f"{c.HEADER}EPOCH %03d" % epoch,
        f"{c.OKBLUE}RL=%4.4f" % stats['rl'][-1],
        f"{c.OKGREEN}KLZ=%4.4f" % stats['klz'][-1],
        f"{c.OKGREEN}KLP=%4.4f" % stats['klp'][-1],
        f"{c.OKBLUE}LOSS=%4.4f" % stats['loss'][-1],
        f"{c.OKCYAN}|rho|=%4.4f{c.ENDC}" % stats['cor'][-1]
    ]
    print(" ".join(to_print))

torch.save({
    'state_dict': vae.state_dict(),
    'stats': stats,
    'args': args,
    'data': data,
}, "trained.model.pth")
