import MDAnalysis as mda
import numpy as np
from kinisi.parser import MDAnalysisParser
from kinisi.diffusion import MSDBootstrap
from tqdm import tqdm

n = snakemake.params['n']
start_diff = snakemake.params['start_diff']

def universe_slice(u: mda.Universe, data_file: str, slice_start: int, slice_end: int) -> mda.Universe:
    """
    Concatenate a list of Universes.

    :param u_list: List of Universe objects
    :param data_file: File path for data
    :param slice_start: Step to start the slice from
    :return: A single flattened Universe
    """
    j = 0
    trajectory = np.zeros((len(u.trajectory[slice_start:slice_end]), len(u.atoms), 3))
    for t in tqdm(u.trajectory[slice_start:slice_end]):
        trajectory[j] = t.positions
        j += 1
    shorter_universe = mda.Universe(data_file, trajectory)
    for t in tqdm(shorter_universe.trajectory):
        for i in shorter_universe.atoms:
            if i.type == 'LI':
                i.type = 'Li'
            if i.type == 'Li1':
                i.type = 'Li'
            if i.type == 'Li2':
                i.type = 'Li'
            if i.type == 'L':
                i.type = 'La'
            if i.type == 'Z':
                i.type = 'Zr'
        t.dimensions = np.array([26.175106662244392197, 26.175106662244392197, 26.175106662244392197, 90, 90, 90])
    return shorter_universe

file = f'src/data/llzo/traj{n}.xyz'


u = mda.Universe(file, file)
uu = universe_slice(u, file, 0, 500)
step_skip = 100  # sampling rate
timestep = 5.079648e-4  # ps
da_params = {'specie': 'Li', 'time_step': timestep, 'step_skip': step_skip}
p = MDAnalysisParser(uu, **da_params)

ngp = np.zeros((8, 4, p.delta_t.size))
no = np.zeros((8, 4, p.delta_t.size))
dt = np.zeros((8, 4, p.delta_t.size))
msd = np.zeros((8, 4, p.delta_t.size))
msd_true = np.zeros((8, 4, p.delta_t.size))
msd_std = np.zeros((8, 4, p.delta_t.size))
cov = np.zeros((8, 4, p.delta_t[np.where(p.delta_t > start_diff)].size, p.delta_t[np.where(p.delta_t > start_diff)].size))
d = np.zeros((8, 4, 3200))
intercept = np.zeros((8, 4, 3200))

for m in range(0, 8, 1):
    for i, slice in enumerate(range(0, 2000, 500)):
        uu = universe_slice(u, file, slice, slice+500)

        rng = np.random.RandomState(42)
        np.random.seed(42)

        da_params = {'specie': 'Li', 'time_step': timestep, 'step_skip': step_skip}
        p = MDAnalysisParser(uu, **da_params)
        n_disp_3d = []
        for t, j in enumerate(p.disp_3d):
            if m == 0:
                n_disp_3d.append(j[::8])
                msd_true[m, i, t] = np.sum(j[::8, ::p.timesteps[t]]**2, axis=-1).mean()
            else:
                n_disp_3d.append(j[m::8])
                msd_true[m, i, t] = np.sum(j[m::8, ::p.timesteps[t]]**2, axis=-1).mean()

        b = MSDBootstrap(p.delta_t, n_disp_3d, p._n_o / 8)
        b.diffusion(**{'dt_skip': start_diff, 'random_state': rng})
        ngp[m, i] = b.ngp
        no[m, i] = p._n_o / 8
        dt[m, i] = b.dt
        msd[m, i] = b.n
        msd_std[m, i] = b.s
        cov[m, i] = b.covariance_matrix
        d[m, i] = b.D.samples
        intercept[m, i] = b.intercept.samples
        
np.savez(f'src/data/llzo/diffusion_{n}_{start_diff}.npz',
        ngp=ngp,
        no=no,
        dt=dt,
        msd_true=msd_true,
        msd=msd,
        msd_std=msd_std,
        cov=cov,
        d=d,
        intercept=intercept)