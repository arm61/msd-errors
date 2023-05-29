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


def nll(*args):
    """
    General purpose negative log-likelihood.
    Returns:
        (:py:attr:`float`): Negative log-likelihood.
    """
    return -log_likelihood(*args)


atoms = int(snakemake.params['atoms'])
length = int(snakemake.params['length'])
jump = int(snakemake.params['jump'])

true_msd = np.load(f'src/data/random_walks/numerical/rw_1_{atoms}_{length}_s4096.npz')['data'][:, 4:]
true_cov = np.cov(true_msd.T)
timestep = np.arange(4, length, 1)

y = true_msd.T
A = np.array([timestep, np.ones(timestep.size)]).T
W = true_cov
V = np.diag(true_cov.diagonal())

g1 = np.matmul(np.linalg.inv(np.matmul(A.T, np.matmul(np.linalg.pinv(W), A))),
               np.matmul(A.T, np.matmul(np.linalg.pinv(W), y)))[0] / 6
g2 = np.matmul(np.linalg.inv(np.matmul(A.T, np.matmul(np.linalg.pinv(V), A))),
               np.matmul(A.T, np.matmul(np.linalg.pinv(V), y)))[0] / 6
g3 = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, y))[0] / 6

W = find_nearest_positive_definite(W)

_, logdet = np.linalg.slogdet(W)
logdet += np.log(2 * np.pi) * true_msd[0].size
inv = pinvh(W)


def log_likelihood(theta: np.ndarray) -> float:
    """
    Get the log likelihood for multivariate normal distribution.
    :param theta: Value of the gradient and intercept of the straight line.
    :return: Log-likelihood value.
    """
    if theta[0] < 0:
        return -np.inf
    model = _straight_line(timestep, *theta)
    diff = (model - true_msd[0])
    logl = -0.5 * (logdet + np.matmul(diff.T, np.matmul(inv, diff)))
    return logl


ols = linregress(timestep, true_msd[0])

max_likelihood = minimize(nll, np.array([ols.slope, ols.intercept])).x
pos = max_likelihood + max_likelihood * 1e-3 * np.random.randn(32, max_likelihood.size)
sampler = EnsembleSampler(*pos.shape, log_likelihood)
sampler.run_mcmc(pos, 1000 + 500, progress=True, progress_kwargs={'desc': "Likelihood Sampling"})
flatchain1 = sampler.get_chain(flat=True, thin=10, discard=500)
flatchain1[:, 0] /= 6

V = find_nearest_positive_definite(V)

_, logdet = np.linalg.slogdet(V)
logdet += np.log(2 * np.pi) * true_msd[0].size
inv = pinvh(V)


def log_likelihood(theta: np.ndarray) -> float:
    """
    Get the log likelihood for multivariate normal distribution.
    :param theta: Value of the gradient and intercept of the straight line.
    :return: Log-likelihood value.
    """
    if theta[0] < 0:
        return -np.inf
    model = _straight_line(timestep, *theta)
    diff = (model - true_msd[0])
    logl = -0.5 * (logdet + np.matmul(diff.T, np.matmul(inv, diff)))
    return logl


max_likelihood = minimize(nll, np.array([ols.slope, ols.intercept])).x
pos = max_likelihood + max_likelihood * 1e-3 * np.random.randn(32, max_likelihood.size)
sampler = EnsembleSampler(*pos.shape, log_likelihood)
sampler.run_mcmc(pos, 1000 + 500, progress=True, progress_kwargs={'desc': "Likelihood Sampling"})
flatchain2 = sampler.get_chain(flat=True, thin=10, discard=500)
flatchain2[:, 0] /= 6


def log_likelihood(theta: np.ndarray) -> float:
    """
    Get the log likelihood for multivariate normal distribution.
    :param theta: Value of the gradient and intercept of the straight line.
    :return: Log-likelihood value.
    """
    model = _straight_line(timestep, *theta)
    diff = (model - true_msd[0])
    U = np.eye(true_msd[0].size) * diff.var()
    U = find_nearest_positive_definite(U)

    _, logdet = np.linalg.slogdet(U)
    logdet += np.log(2 * np.pi) * true_msd[0].size
    inv = pinvh(U)
    if theta[0] < 0:
        return -np.inf
    logl = -0.5 * (logdet + np.matmul(diff.T, np.matmul(inv, diff)))
    return logl


max_likelihood = minimize(nll, np.array([ols.slope, ols.intercept])).x
pos = max_likelihood + max_likelihood * 1e-3 * np.random.randn(32, max_likelihood.size)
sampler = EnsembleSampler(*pos.shape, log_likelihood)
sampler.run_mcmc(pos, 1000 + 500, progress=True, progress_kwargs={'desc': "Likelihood Sampling"})
flatchain3 = sampler.get_chain(flat=True, thin=10, discard=500)
flatchain3[:, 0] /= 6

np.savez(f'src/data/random_walks/numerical/glswlsols_1_{atoms}_{length}.npz',
         gls_pop=g1,
         wls_pop=g2,
         ols_pop=g3,
         gls_est=flatchain1,
         wls_est=flatchain2,
         ols_est=flatchain3)
