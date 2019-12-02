import matplotlib.pyplot as plt
import numpy as np
import torch

from train_old import train_old
from train import train


params = {
    'seed': 1234,
    'lr': 2e-2,
    'batch_size': 20,
    'training_batches': 200,
}

history_loss_euclidean = train_old(params, gradient_type='euclidean')
history_loss_quasi_diagonal_natural = train_old(params, gradient_type='quasi-diagonal-natural')
history_loss_euclidean_new = train(params, gradient_type='euclidean')
history_loss_quasi_diagonal_natural_new = train(params, gradient_type='quasi-diagonal-natural')

plt.plot(history_loss_euclidean, label='euclidean')
plt.plot(history_loss_quasi_diagonal_natural, label='quasi-diagonal natural')
plt.plot(history_loss_euclidean_new, label='euclidean new', ls='--')
plt.plot(history_loss_quasi_diagonal_natural_new, label='quasi-diagonal natural new', ls='--')
plt.yscale('log')
plt.ylim(1e-5, 1e2)
plt.legend()
plt.savefig('natural_grad.pdf')
