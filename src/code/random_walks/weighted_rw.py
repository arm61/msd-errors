# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

import numpy as np
from scipy.stats import linregress
from scipy.linalg import pinvh
from scipy.optimize import minimize
from emcee import EnsembleSampler
from kinisi.diffusion import _straight_line
from tqdm import tqdm

jump = snakemake.params['jump']
atoms = snakemake.params['atoms']
length = int(snakemake.params['length'])
size = int(snakemake.params['n'])

gradient = np.zeros((size, 3200))
intercept = np.zeros((size, 3200))
data = np.load(f'src/data/random_walks/kinisi/rw_1_{atoms}_{length}_s{size}.npz')

for seed in tqdm(range(size)):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    x = data['data'][seed, 4:, 0]
    y = data['data'][seed, 4:, 1]
    yerr = data['data'][seed, 4:, 2]  

    rcov = np.diag(yerr ** 2)

    _, logdet = np.linalg.slogdet(rcov)
    logdet += np.log(2 * np.pi) * y.size
    inv = pinvh(rcov)

    def log_likelihood(theta: np.ndarray) -> float:
        """
        Get the log likelihood for multivariate normal distribution.
        :param theta: Value of the gradient and intercept of the straight line.
        :return: Log-likelihood value.
        """
        if theta[0] < 0:
            return -np.inf
        model = _straight_line(x, *theta)
        diff = (model - y)
        logl = -0.5 * (logdet + np.matmul(diff.T, np.matmul(inv, diff)))
        return logl

    ols = linregress(x, y)
    slope = ols.slope
    c = 1e-20
    if slope < 0:
        slope = 1e-20

    def nll(*args) -> float:
        """
        General purpose negative log-likelihood.
        :return: Negative log-likelihood
        """
        return -log_likelihood(*args)

    max_likelihood = minimize(nll, np.array([slope, c])).x
    pos = max_likelihood + max_likelihood * 1e-3 * np.random.randn(32, max_likelihood.size)
    sampler = EnsembleSampler(*pos.shape, log_likelihood)
    sampler.run_mcmc(pos, 1000 + 500, progress=False)
    flatchain = sampler.get_chain(flat=True, thin=10, discard=500)
    gradient[seed] = flatchain[:, 0] / 6
    intercept[seed] = flatchain[:, 1]

np.savez(f'src/data/random_walks/weighted/rw_1_{atoms}_{length}_s{size}.npz',
         diff_c=gradient,
         intercept=intercept)
