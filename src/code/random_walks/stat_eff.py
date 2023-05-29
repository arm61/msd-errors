import numpy as np
from scipy.optimize import curve_fit

jump = 1

xaxis = np.array([16, 32, 64, 128, 256, 512, 1024])

kinisi_atoms = np.zeros((xaxis.size))
weighted_atoms = np.zeros((xaxis.size))
ordinary_atoms = np.zeros((xaxis.size))
numerical_atoms = np.zeros((xaxis.size))
kinisi_length = np.zeros((xaxis.size))
weighted_length = np.zeros((xaxis.size))
ordinary_length = np.zeros((xaxis.size))
numerical_length = np.zeros((xaxis.size))
for i, ii in enumerate(xaxis):
    length = 128
    file_open = np.load(f"src/data/random_walks/kinisi/rw_{jump}_{ii}_{length}_s512.npz")
    kinisi_atoms[i] = file_open['diff_c'].mean(-1).var(-1, ddof=1)
    file_open = np.load(f"src/data/random_walks/weighted/rw_{jump}_{ii}_{length}_s512.npz")
    weighted_atoms[i] = file_open['diff_c'].mean(-1).var(-1, ddof=1)
    file_open = np.load(f"src/data/random_walks/ordinary/rw_{jump}_{ii}_{length}_s512.npz")
    ordinary_atoms[i] = file_open['diff_c'].mean(-1).var(-1, ddof=1)
    file_open = np.load(f"src/data/random_walks/numerical/D_{jump}_{ii}_{length}.npz")
    numerical_atoms[i] = file_open['diff_c'].var(ddof=1)
for i, ii in enumerate(xaxis):
    atoms = 128
    file_open = np.load(f"src/data/random_walks/kinisi/rw_{jump}_{atoms}_{ii}_s512.npz")
    kinisi_length[i] = file_open['diff_c'].mean(-1).var(-1, ddof=1)
    file_open = np.load(f"src/data/random_walks/weighted/rw_{jump}_{atoms}_{ii}_s512.npz")
    weighted_length[i] = file_open['diff_c'].mean(-1).var(-1, ddof=1)
    file_open = np.load(f"src/data/random_walks/ordinary/rw_{jump}_{atoms}_{ii}_s512.npz")
    ordinary_length[i] = file_open['diff_c'].mean(-1).var(-1, ddof=1)
    file_open = np.load(f"src/data/random_walks/numerical/D_{jump}_{atoms}_{ii}.npz")
    numerical_length[i] = file_open['diff_c'].var(ddof=1)


def f(x, a, b):
    return b * x**a


kopt, kcov = curve_fit(f, xaxis, kinisi_atoms, sigma=kinisi_atoms)
wopt, wcov = curve_fit(f, xaxis, weighted_atoms, sigma=weighted_atoms)
popt, pcov = curve_fit(f, xaxis, ordinary_atoms, sigma=ordinary_atoms)
nopt, ncov = curve_fit(f, xaxis, numerical_atoms, sigma=numerical_atoms)
kerr = np.sqrt(np.diagonal(kcov))
werr = np.sqrt(np.diagonal(wcov))
perr = np.sqrt(np.diagonal(pcov))
nerr = np.sqrt(np.diagonal(ncov))

kopt2, kcov2 = curve_fit(f, xaxis, kinisi_length, sigma=kinisi_length)
wopt2, wcov2 = curve_fit(f, xaxis, weighted_length, sigma=weighted_length)
popt2, pcov2 = curve_fit(f, xaxis, ordinary_length, sigma=ordinary_length)
nopt2, ncov2 = curve_fit(f, xaxis, numerical_length, sigma=numerical_length)
kerr2 = np.sqrt(np.diagonal(kcov2))
werr2 = np.sqrt(np.diagonal(wcov2))
perr2 = np.sqrt(np.diagonal(pcov2))
nerr2 = np.sqrt(np.diagonal(ncov2))

np.savez(f"src/data/random_walks/stat_eff.npz",
         x=np.log2(xaxis),
         ka=kopt[0],
         kb=kopt[1],
         dka=kerr[0],
         dkb=kerr[1],
         ky=kinisi_atoms,
         wa=wopt[0],
         wb=wopt[1],
         dwa=werr[0],
         dwb=werr[1],
         wy=weighted_atoms,
         pa=popt[0],
         pb=popt[1],
         dpa=perr[0],
         dpb=perr[1],
         py=ordinary_atoms,
         na=nopt[0],
         nb=nopt[1],
         dna=nerr[0],
         dnb=nerr[1],
         ny=numerical_atoms,
         ka2=kopt2[0],
         kb2=kopt2[1],
         dka2=kerr2[0],
         dkb2=kerr2[1],
         ky2=kinisi_length,
         wa2=wopt2[0],
         wb2=wopt2[1],
         dwa2=werr2[0],
         dwb2=werr2[1],
         wy2=weighted_length,
         pa2=popt2[0],
         pb2=popt2[1],
         dpa2=perr2[0],
         dpb2=perr2[1],
         py2=ordinary_length,
         na2=nopt2[0],
         nb2=nopt2[1],
         dna2=nerr2[0],
         dnb2=nerr2[1],
         ny2=numerical_length)
