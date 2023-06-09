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

true_msd = np.load(f'src/data/random_walks/numerical/rw_1_{atoms}_{length}_s4096.npz')['data'][:, 4:]
true_cov = np.cov(true_msd.T)
timestep = np.arange(4, length, 1)

y = true_msd.T
A = np.array([timestep, np.ones(timestep.size)]).T
W = true_cov
V = np.diag(true_cov.diagonal())

g1 = np.matmul(np.linalg.inv(np.matmul(A.T, np.matmul(np.linalg.pinv(W), A))),
               np.matmul(A.T, np.matmul(np.linalg.pinv(W), y)))[0] / 6
c1 = np.sqrt(np.linalg.inv(np.matmul(A.T, np.matmul(np.linalg.pinv(W), A)))[0][0]) / 6
g2 = np.matmul(np.linalg.inv(np.matmul(A.T, np.matmul(np.linalg.pinv(V), A))),
               np.matmul(A.T, np.matmul(np.linalg.pinv(V), y)))[0] / 6
c2 = np.sqrt(np.linalg.inv(np.matmul(A.T, np.matmul(np.linalg.pinv(V), A)))[0][0]) / 6
g3 = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, y))[0] / 6

c3 = []
for i in true_msd:
    c3.append(linregress(timestep, i).stderr / 6)

np.savez(f'src/data/random_walks/numerical/glswlsols_1_{atoms}_{length}.npz',
         gls_pop=g1,
         wls_pop=g2,
         ols_pop=g3,
         gls_est=c1,
         wls_est=c2,
         ols_est=np.mean(c3))
