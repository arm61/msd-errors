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

gradient = np.zeros((size))
gradient_err = np.zeros((size))
intercept = np.zeros((size))
intercept_err = np.zeros((size))
data = np.load(f'src/data/random_walks/kinisi/rw_1_{atoms}_{length}_s{size}.npz')

V = np.diag(data['data'][:, 4:, 1].var(0))

for seed in tqdm(range(size)):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    A = np.array([data['data'][seed, 4:, 0], np.ones(data['data'][seed, 4:, 0].size)]).T
    y = data['data'][seed, 4:, 1].T

    g2 = np.matmul(np.linalg.inv(np.matmul(A.T, np.matmul(np.linalg.pinv(V), A))),
                np.matmul(A.T, np.matmul(np.linalg.pinv(V), y)))
    c2 = np.sqrt(np.linalg.inv(np.matmul(A.T, np.matmul(np.linalg.pinv(V), A))))

    gradient[seed] =  g2[0] / 6
    gradient_err[seed] =  c2[0, 0] / 6
    intercept[seed] = g2[1]
    intercept_err[seed] = c2[1, 1]

np.savez(f'src/data/random_walks/weighted/rw_1_{atoms}_{length}_s{size}.npz',
         diff_c=gradient,
         intercept=intercept,
         diff_c_err=gradient_err,
         intercept_err=intercept_err)
