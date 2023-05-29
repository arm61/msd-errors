# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

import numpy as np
from scipy.stats import linregress
from scipy.linalg import pinvh
from emcee import EnsembleSampler
from scipy.optimize import minimize
from kinisi.matrix import find_nearest_positive_definite
from kinisi.diffusion import _straight_line

np.random.seed(1)

atoms = int(snakemake.params['atoms'])
length = int(snakemake.params['length'])
jump = int(snakemake.params['jump'])

true_msd = np.load(f'src/data/random_walks/numerical/rw_1_{atoms}_{length}_s4096.npz')['data']
timestep = np.arange(1, length + 1, 1, dtype=int)
true_mean = true_msd.mean(0)
rcov = np.cov(true_msd.T)

max_ngp = np.argwhere(timestep > 4)[0][0]
rcov = find_nearest_positive_definite(rcov[max_ngp:, max_ngp:])

_, logdet = np.linalg.slogdet(rcov)
logdet += np.log(2 * np.pi) * true_msd.mean(0)[max_ngp:].size
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
    diff = (model - true_msd.mean(0)[max_ngp:])
    logl = -0.5 * (logdet + np.matmul(diff.T, np.matmul(inv, diff)))
    return logl


ols = linregress(timestep[max_ngp:], true_msd.mean(0)[max_ngp:])


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
np.savez(f'src/data/random_walks/numerical/D_1_{atoms}_{length}.npz',
         diff_c=flatchain[:, 0] / 6,
         intercept=flatchain[:, 1])
