# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey (arm61)

from typing import Union, List
import numpy as np


def walk(atoms: int,
         timesteps: np.ndarray,
         jump_size: int = 1,
         seed: Union[np.random.mtrand.RandomState, None] = None) -> np.ndarray:
    """
    Perform a random walk.

    :param atoms: number of atoms
    :param timesteps: the timestep values
    :param jump_size: size of jump
    :param seed: random seed source
    :return: cumulative sum of steps for walk
    """
    possible_moves = np.zeros((6, 3))
    j = 0
    for i in range(0, 6, 2):
        possible_moves[i, j] = jump_size
        possible_moves[i + 1, j] = -jump_size
        j += 1
    choices = seed.choice(len(range(len(possible_moves))), size=(atoms, len(timesteps)))
    steps = np.zeros((atoms, len(timesteps), 3))
    for i in range(steps.shape[0]):
        for j in range(steps.shape[1]):
            steps[i, j] = possible_moves[choices[i, j]]
    cum_steps = np.cumsum(steps, axis=1)
    return cum_steps


def get_disp3d(which_walk: callable,
               steps: int,
               atoms: int,
               jump_size: int = 1,
               seed: Union[np.random.mtrand.RandomState, None] = None) -> List[np.ndarray]:
    """
    Return the three-dimensional displacements from a given random walk.

    :param which_walk: the walk to perform.
    :param steps: number of timesteps.
    :param atoms: number of atoms.
    :param jump_size: size of jump
    :param seed: random seed source
    :return: Three-dimensional displacements
    """
    dt = np.linspace(1, steps, steps, dtype=int)
    cum_steps = which_walk(atoms, dt, jump_size=jump_size, seed=seed)
    disp_3d = []
    n_samples = np.array([])
    for i, n in enumerate(dt):
        disp = np.concatenate([cum_steps[:, np.newaxis, i],
                               np.subtract(cum_steps[:, i + 1:], cum_steps[:, :-(i + 1)])],
                              axis=1)
        disp_3d.append(disp)
        n_samples = np.append(n_samples, dt[-1] / n * atoms)
    return disp_3d, n_samples
