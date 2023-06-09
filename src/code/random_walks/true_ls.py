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
from tqdm import tqdm

np.random.seed(1)

atoms = int(snakemake.params['atoms'])
length = int(snakemake.params['length'])
jump = int(snakemake.params['jump'])
size = 4096

true_msd = np.load(f'src/data/random_walks/numerical/rw_1_{atoms}_{length}_s{size}.npz')['data'][:, 4:]
timestep = np.arange(1, length + 1, 1, dtype=int)[4:]
true_mean = true_msd.mean(0)
W = np.cov(true_msd.T)

gradient = np.zeros((size))
gradient_err = np.zeros((size))
intercept = np.zeros((size))
intercept_err = np.zeros((size))

for seed in tqdm(range(size)):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    A = np.array([timestep, np.ones(timestep.size)]).T
    y = true_msd[seed].T
    g1 = np.matmul(np.linalg.inv(np.matmul(A.T, np.matmul(np.linalg.pinv(W), A))),
                np.matmul(A.T, np.matmul(np.linalg.pinv(W), y)))
    c1 = np.sqrt(np.linalg.inv(np.matmul(A.T, np.matmul(np.linalg.pinv(W), A))))

    gradient[seed] =  g1[0] / 6
    gradient_err[seed] =  c1[0, 0] / 6
    intercept[seed] = g1[1]
    intercept_err[seed] = c1[1, 1]

np.savez(f'src/data/random_walks/numerical/D_1_{atoms}_{length}.npz',
         diff_c=gradient,
         intercept=intercept,
         diff_c_err=gradient_err,
         intercept_err=intercept_err)
