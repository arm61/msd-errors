# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

import numpy as np
from sklearn.utils import resample
from tqdm import tqdm
from uravu.distribution import Distribution
from random_walk import walk, anticorrelated_walk, get_disp3d
from kinisi.diffusion import MSDBootstrap

jump = snakemake.params['jump']
atoms = snakemake.params['atoms']
length = int(snakemake.params['length'])
correlated = snakemake.params['correlation']
size = int(snakemake.params['n'])

timestep = np.linspace(1, length, length, dtype=int)
res = np.zeros((size, length - 2))
res_e = np.zeros((size, length - 2))
for seed in tqdm(range(size)):
    data = np.zeros((length - 2, 3))

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    if correlated == 'correlated':
        disp_3d = get_disp3d(anticorrelated_walk, length, atoms, seed=rng)
    else:
        disp_3d = get_disp3d(walk, length, atoms, seed=rng)

    diff = MSDBootstrap(timestep, disp_3d, random_state=rng, progress=False)

    i = 0
    for n in range(len(diff.dt))[1:]:
        r2_sum = np.sum(diff._displacements[n]**2, axis=2)
        resampled = [np.mean(resample(r2_sum.flatten())) for j in range(1000)]
        distro = Distribution(resampled)
        while (not distro.normal):
            distro.add_samples([np.mean(resample(r2_sum.flatten())) for j in range(100)])
        res[seed, i] = distro.n
        res_e[seed, i] = np.var(distro.samples, ddof=1)
        i += 1

np.savez(f'src/data/random_walks/nonindependent/{correlated}/rw_{int(jump)}_{atoms}_{length}_s{size}.npz',
         res_msd=res,
         res_var=res_e)
