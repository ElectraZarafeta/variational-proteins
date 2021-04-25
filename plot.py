import torch
import matplotlib.pyplot as plt
import os
from vae_bayesian import VAE
import numpy as np
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model_dict = torch.load('trained.model.pth')

plt.figure(figsize=(35,4))
plt.subplot(1,4,1)
plt.title("Reconstruction Loss (RL)")
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.set_xlabel('EPOCH', c='C3')
ax1.tick_params(axis='x', labelcolor='C3')
ax1.set_ylabel('Reconstruction Loss (RL)', c='C0')
ax1.tick_params(axis='y', labelcolor='C0')
ax1.plot(model_dict['stats']['rl'], lw=2, c='C0')

plt.subplot(1,4,2)
plt.title("Kullback-Leibler divergence loss (KL) for parameters")
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.set_xlabel('EPOCH', c='C3')
ax1.tick_params(axis='x', labelcolor='C3')
ax1.set_ylabel('Kullback-Leibler divergence loss (KL) for parameters', c='C0')
ax1.tick_params(axis='y', labelcolor='C0')
ax1.plot(model_dict['stats']['klp'], lw=2, c='C0')

plt.subplot(1,4,3)
plt.title("Kullback-Leibler divergence loss (KL) for z")
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.set_xlabel('EPOCH', c='C3')
ax1.tick_params(axis='x', labelcolor='C3')
ax1.set_ylabel('Kullback-Leibler divergence loss (KL) for z', c='C0')
ax1.tick_params(axis='y', labelcolor='C0')
ax1.plot(model_dict['stats']['klz'], lw=2, c='C0')

plt.subplot(1,4,4)
plt.title("Total Loss")
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.set_xlabel('EPOCH', c='C3')
ax1.tick_params(axis='x', labelcolor='C3')
ax1.set_ylabel('Total loss', c='C0')
ax1.tick_params(axis='y', labelcolor='C0')
ax1.plot(model_dict['stats']['loss'], lw=2, c='C0')
plt.show()

"""
plt.figure(figsize=(8,4))
plt.title(r"$|Spearman\ \rho|$ correlation to experimental data")
plt.xlabel('EPOCH', c='C3')
plt.tick_params(axis='x', labelcolor='C3')
plt.plot(model_dict['stats']['cor'], lw=2, c='C9', label="Our result")
plt.tick_params(axis='y', labelcolor='C9')
plt.axhline(y=0.74388, c='C6', lw=2, label=f'Paper result (without ensambling) ' + rf'$|\rho|={round(0.74388, 4)}$')
plt.legend()
plt.show()
"""

plt.figure(figsize=(8,6))
plt.title("Latent space")
mask = model_dict['data']['df']['label'].isin(model_dict['data']['df']['label'].value_counts()[:5].index) # We limit to top 5 classes only
vae = VAE(**model_dict['args'])
vae.eval()
_, mu, logvar = vae(model_dict['data']['dataloader'].dataset[mask], rep=False)
columns = [str(i+1) for i in range(mu.shape[1])] + ['label']
dfp = pd.DataFrame(data=np.c_[mu.detach().numpy(), model_dict['data']['df'][mask]['label']], columns=columns)
dfp = dfp.set_index('1').groupby('label')['2']
dfp.plot(style='.', ms=2, alpha=0.5, legend=True)
plt.show()