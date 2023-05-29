# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

import numpy as np
from tqdm import tqdm
from random_walk import walk

jump = snakemake.params['jump']
atoms = snakemake.params['atoms']
length = int(snakemake.params['length'])
size = int(snakemake.params['n'])

timestep = np.arange(1, length + 1, 1, dtype=int)
k = np.zeros((size, timestep.size))
for seed in tqdm(range(size)):
    rng = np.random.RandomState(seed)

    cum_steps = walk(atoms, timestep, seed=rng, jump_size=jump)

    for i, n in enumerate(timestep):
        disp = np.concatenate([cum_steps[:, np.newaxis, i],
                               np.subtract(cum_steps[:, i + 1:], cum_steps[:, :-(i + 1)])],
                              axis=1)
        d2 = np.sum(disp**2, axis=-1)
        k[seed, i] = np.mean(d2.flatten())

np.savez(f'src/data/random_walks/numerical/rw_1_{atoms}_{length}_s{size}.npz', data=k)
