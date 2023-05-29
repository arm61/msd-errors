# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

import numpy as np
from scipy.stats import multivariate_normal, linregress
from scipy.optimize import minimize
from emcee import EnsembleSampler
from kinisi.matrix import find_nearest_positive_definite
from uravu.utils import straight_line

atoms = 1024
lengths = [128, 256, 52, 1024, 2048, 4096]
jumps = [1, 8, 32]

data = np.zeros((2, len(jumps), len(lengths), 3200))

for j, jump in enumerate(jumps):
    for k, length in enumerate(lengths):
        simulation = np.load(f'../data/numerical_{jump}_{atoms}_{length}.npz')['data']
        covariance = np.cov(simulation.T)
        timestep = np.linspace(1, length, length, dtype=int)[1:-1]

        covariance = find_nearest_positive_definite(covariance)
        mv = multivariate_normal(simulation.mean(0), covariance, allow_singular=True)

        def log_likelihood(theta):
            """
            Get the log likelihood for multivariate normal distribution.
            Args:
                theta (:py:attr:`array_like`): Value of the gradient and intercept of the straight line.
            Returns:
                (:py:attr:`float`): Log-likelihood value.
            """
            model = straight_line(timestep, *theta)
            logl = mv.logpdf(model)
            return logl

        ols = linregress(timestep, simulation.mean(0))

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
        sampler.run_mcmc(pos, 1500, progress=True, progress_kwargs={'desc': "Likelihood Sampling"})
        flatchain = sampler.get_chain(flat=True, discard=500)
        data[0, j, k] = flatchain[:, 0] / 6
        data[1, j, k] = flatchain[:, 1]
np.save('src/data/longtime.npy', data)
