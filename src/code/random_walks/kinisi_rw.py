# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

import numpy as np
from kinisi.diffusion import MSDBootstrap
from tqdm import tqdm
from random_walk import get_disp3d, walk

jump = snakemake.params['jump']
atoms = snakemake.params['atoms']
length = int(snakemake.params['length'])
size = int(snakemake.params['n'])

timestep = np.arange(1, length + 1, 1, dtype=int)
data = np.zeros((size, timestep.size, 4))
covariance = np.zeros((size, timestep.size - 4, timestep.size - 4))
npd_covariance = np.zeros((size, timestep.size - 4, timestep.size - 4))
n_o = np.zeros((size, timestep.size))
diff_c = np.zeros((size, 3200))
intercept = np.zeros((size, 3200))

for seed in tqdm(range(size)):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    disp_3d, n_samples = get_disp3d(walk, length, atoms, jump_size=jump, seed=rng)

    diff = MSDBootstrap(timestep, disp_3d, n_samples, random_state=rng, progress=False)
    diff.diffusion(use_ngp=False, random_state=rng, dt_skip=4, progress=False)
    diff_c[seed] = diff.gradient.samples / 6
    intercept[seed] = diff.intercept.samples
    data[seed, :, 0] = diff.dt
    data[seed, :, 1] = diff.n
    data[seed, :, 2] = diff.s
    data[seed, 4:, 3] = diff._model_v
    npd_covariance[seed] = diff._npd_covariance_matrix
    covariance[seed] = diff.covariance_matrix
    n_o[seed] = diff._n_o

np.savez(f'src/data/random_walks/kinisi/rw_1_{atoms}_{length}_s{size}.npz',
         diff_c=diff_c,
         intercept=intercept,
         data=data,
         covariance=covariance,
         npd_covariance=npd_covariance,
         n_o=n_o)
