# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

import numpy as np
from scipy.stats import linregress
from scipy.linalg import pinvh
from emcee import EnsembleSampler
from scipy.optimize import minimize
from kinisi.diffusion import _straight_line

np.random.seed(1)

start_diff = snakemake.params['start_diff']
length = snakemake.params['length']

timestep = np.load(f'src/data/llzo/diffusion_0_{start_diff}_{length}.npz')['dt'][0, 0]
ll = len([i + length for i in range(0, 20001 - length, length)])
# length = 500 
# ll = len([i + length for i in range(0, 2000, length)])
d = np.zeros((5, 16, ll, timestep.size))
for i in range(1, 6, 1):
    d[i-1] = np.load(f'src/data/llzo/diffusion_{i}_{start_diff}_{length}.npz')['msd_true']

true_mean = d.reshape(-1, d.shape[-1]).mean(0)
max_ngp = np.argwhere(timestep > start_diff)[0][0]
rcov = np.cov(d.reshape(-1, d.shape[-1]).T)[max_ngp:, max_ngp:]

_, logdet = np.linalg.slogdet(rcov)
logdet += np.log(2 * np.pi) * d.reshape(-1, d.shape[-1]).mean(0)[max_ngp:].size
inv = pinvh(rcov)

def log_likelihood(theta: np.ndarray) -> float:
    """
    Get the log likelihood for multivariate normal distribution.
    :param theta: Value of the gradient and intercept of the straight line.
    :return: Log-likelihood value.
    """
    if theta[0] < 0:
        return -np.inf
    model = _straight_line(timestep[max_ngp:], *theta)
    diff = (model - d.reshape(-1, d.shape[-1]).mean(0)[max_ngp:])
    logl = -0.5 * (logdet + np.matmul(diff.T, np.matmul(inv, diff)))
    return logl

ols = linregress(timestep[max_ngp:], d.reshape(-1, d.shape[-1]).mean(0)[max_ngp:])


def nll(*args):
    """
    General purpose negative log-likelihood.
    Returns:
        (:py:attr:`float`): Negative log-likelihood.
    """
    return -log_likelihood(*args)


max_likelihood = minimize(nll, np.array([ols.slope, ols.intercept])).x
pos = max_likelihood + max_likelihood * 1e-3 * np.random.randn(32, max_likelihood.size)
sampler = EnsembleSampler(*pos.shape, log_likelihood)
sampler.run_mcmc(pos, 1000 + 500, progress=True, progress_kwargs={'desc': "Likelihood Sampling"})
flatchain = sampler.get_chain(flat=True, thin=10, discard=500)
np.savez(f'src/data/llzo/true_{start_diff}_{length}.npz',
         diff_c=flatchain[:, 0] / 6e4,
         intercept=flatchain[:, 1])
