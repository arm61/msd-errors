import MDAnalysis as mda
import numpy as np
from kinisi.parser import MDAnalysisParser
from kinisi.diffusion import MSDBootstrap

n = snakemake.params['n']
start_diff = snakemake.params['start_diff']
length = snakemake.params['length']

step = 16
n_atoms_drift = int((1536-448) / step)
n_atoms_li = int(448 / step)
rng = np.random.RandomState(n*42)
np.random.seed(n*42)
choice_drift = rng.choice(np.arange(0, 1536-448, dtype=int), size=(step, n_atoms_drift), replace=False)
choice_li = rng.choice(np.arange(1536-448, 1536, dtype=int), size=(step, n_atoms_li), replace=False)
choice = np.concatenate([choice_drift, choice_li], axis=-1)

def universe_slice(data: str, slice_start: int, slice_end: int, m: int) -> mda.Universe:
    """
    Create slice universe.

    :param data: Numpy data of coordinates
    :param slice_start: Step to start the slice from
    :param slice_end: Step to end the slice from
    :return: A single flattened Universe
    """
    data_reshape = np.copy(data).reshape(-1, 1536, 3)
    data_subset = data_reshape[slice_start:slice_end, choice_li[m]]
    data_subset *= 0.52917721067
    u = mda.Universe.empty(data_subset.shape[1], 1, n_frames=data_subset.shape[0], trajectory=True)
    list_type = np.array(['Li'] * n_atoms_li)
    u.add_TopologyAttr('type', list(list_type))
    for i, t in enumerate(u.trajectory):
        t.positions = data_subset[i]
        t.dimensions = np.array([26.30938114, 26.30938114, 26.30938114, 90, 90, 90])
    return u

data = np.loadtxt(f'src/data/llzo/traj{n}.out', usecols=[0, 1, 2])
uu = universe_slice(data, 0, length, 0)
step_skip = 100  # sampling rate
timestep = 5.079648e-4  # ps
da_params = {'specie': 'Li', 'time_step': timestep, 'step_skip': step_skip}
p = MDAnalysisParser(uu, **da_params)

ll = len([i + length for i in range(0, 20001 - length, length)])
dt = np.zeros((step, ll, p.delta_t.size))
msd = np.zeros((step, ll, p.delta_t.size))
msd_true = np.zeros((step, ll, p.delta_t.size))
msd_std = np.zeros((step, ll, p.delta_t.size))
n_o = np.zeros((step, ll, p.delta_t.size))
cov = np.zeros((step, ll, p.delta_t[np.where(p.delta_t > start_diff)].size, p.delta_t[np.where(p.delta_t > start_diff)].size))
d = np.zeros((step, ll, 3200))
g = np.zeros((step, ll, 3200))
intercept = np.zeros((step, ll, 3200))

for m in range(0, step, 1):
    for i, slice in enumerate(range(0, 20001 - length, length)):
        uu = universe_slice(data, slice, slice+length, m)

        da_params = {'specie': 'Li', 'time_step': timestep, 'step_skip': step_skip, 'progress': False}
        p = MDAnalysisParser(uu, **da_params)
        b = MSDBootstrap(p.delta_t, p.disp_3d, p._n_o, progress=False)
        b.diffusion(start_diff, random_state=rng, progress=False)
        dt[m, i] = b.dt
        msd[m, i] = b.n
        msd_true[m, i] = b.n
        msd_std[m, i] = b.s
        n_o[m, i] = p._n_o
        cov[m, i] = b.covariance_matrix
        d[m, i] = b.D.samples
        g[m, i] = b.gradient.samples
        intercept[m, i] = b.intercept.samples

np.savez(f'src/data/llzo/diffusion_{n}_{start_diff}_{length}.npz',
        dt=dt,
        msd_true=msd_true,
        msd=msd,
        msd_std=msd_std,
        n_o = n_o,
        cov=cov,
        d=d,
        g=g,
        intercept=intercept)