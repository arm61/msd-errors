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
data = np.load(f'src/data/random_walks/kinisi//rw_1_{atoms}_{length}_s{size}.npz')['data']

for seed in tqdm(range(size)):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    x = data[seed, 4:, 0]
    y = data[seed, 4:, 1]
    lin = linregress(x, y)
    
    gradient[seed] =  lin.slope / 6
    gradient_err[seed] =  lin.stderr / 6
    intercept[seed] = lin.intercept
    intercept_err[seed] = lin.intercept_stderr

np.savez(f'src/data/random_walks/ordinary/rw_1_{atoms}_{length}_s{size}.npz',
         diff_c=gradient,
         intercept=intercept,
         diff_c_err=gradient_err,
         intercept_err=intercept_err)