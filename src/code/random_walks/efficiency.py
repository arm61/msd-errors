import numpy as np
from scipy.optimize import curve_fit
from uravu.distribution import Distribution
from uravu.relationship import Relationship

jump = 1

xaxis = np.array([16, 32, 64, 128, 256, 512, 1024])

kinisi_atoms = np.zeros((xaxis.size, 512, 3200))
weighted_atoms = np.zeros((xaxis.size, 512, 3200))
ordinary_atoms = np.zeros((xaxis.size, 512, 3200))
kinisi_length = np.zeros((xaxis.size, 512, 3200))
weighted_length = np.zeros((xaxis.size, 512, 3200))
ordinary_length = np.zeros((xaxis.size, 512, 3200))
true_d_atoms = np.zeros((xaxis.size, 3200))
true_d_length = np.zeros((xaxis.size, 3200))
for i, ii in enumerate(xaxis):
    length = 128
    true_d_atoms[i] = np.load(f"src/data/random_walks/numerical/D_{jump}_{ii}_{length}.npz")["diff_c"]
    file_open = np.load(f"src/data/random_walks/kinisi/rw_{jump}_{ii}_{length}_s512.npz")
    kinisi_atoms[i] = file_open['diff_c']
    file_open = np.load(f"src/data/random_walks/weighted/rw_{jump}_{ii}_{length}_s512.npz")
    weighted_atoms[i] = file_open['diff_c']
    file_open = np.load(f"src/data/random_walks/ordinary/rw_{jump}_{ii}_{length}_s512.npz")
    ordinary_atoms[i] = file_open['diff_c']
for i, ii in enumerate(xaxis):
    atoms = 128
    true_d_length[i] = np.load(f"src/data/random_walks/numerical/D_{jump}_{atoms}_{ii}.npz")["diff_c"]
    file_open = np.load(f"src/data/random_walks/kinisi/rw_{jump}_{atoms}_{ii}_s512.npz")
    kinisi_length[i] = file_open['diff_c']
    file_open = np.load(f"src/data/random_walks/weighted/rw_{jump}_{atoms}_{ii}_s512.npz")
    weighted_length[i] = file_open['diff_c']
    file_open = np.load(f"src/data/random_walks/ordinary/rw_{jump}_{atoms}_{ii}_s512.npz")
    ordinary_length[i] = file_open['diff_c']

print(np.log2(true_d_length.var(-1, ddof=1)))


def f(x, a, b):
    return b * x**a


ky = [Distribution(i.var(-1, ddof=1)) for i in kinisi_atoms]
wy = [Distribution(i.var(-1, ddof=1)) for i in weighted_atoms]
py = [Distribution(i.var(-1, ddof=1)) for i in ordinary_atoms]
kr = Relationship(f, xaxis, ky, bounds=((-3.5, 0.5), (0, 1)))
wr = Relationship(f, xaxis, wy, bounds=((-3.5, 0.5), (0, 1)))
pr = Relationship(f, xaxis, py, bounds=((-3.5, 0.5), (0, 1)))
kr.max_likelihood('diff_evo')
wr.max_likelihood('diff_evo')
pr.max_likelihood('diff_evo')
kr.mcmc()
wr.mcmc()
pr.mcmc()
true, pcov = curve_fit(f, xaxis, true_d_atoms.var(-1, ddof=1))
print([np.mean(i.samples) for i in kr.variables])
print([np.mean(i.samples) for i in wr.variables])
print([np.mean(i.samples) for i in pr.variables])
print(true)
perr = np.sqrt(np.diagonal(pcov))

ky2 = [Distribution(i.var(-1, ddof=1)) for i in kinisi_length]
wy2 = [Distribution(i.var(-1, ddof=1)) for i in weighted_length]
py2 = [Distribution(i.var(-1, ddof=1)) for i in ordinary_length]
kr2 = Relationship(f, xaxis, ky2, bounds=((-3.5, 0.5), (0, 1)))
wr2 = Relationship(f, xaxis, wy2, bounds=((-3.5, 0.5), (0, 1)))
pr2 = Relationship(f, xaxis, py2, bounds=((-3.5, 0.5), (0, 1)))
kr2.max_likelihood('diff_evo')
wr2.max_likelihood('diff_evo')
pr2.max_likelihood('diff_evo')
kr2.mcmc()
wr2.mcmc()
pr2.mcmc()
true2, pcov = curve_fit(f, xaxis, true_d_length.var(-1, ddof=1))
print([np.mean(i.samples) for i in kr2.variables])
print([np.mean(i.samples) for i in wr2.variables])
print([np.mean(i.samples) for i in pr2.variables])
print(true2)
perr2 = np.sqrt(np.diagonal(pcov))
np.savez(f"src/data/random_walks/efficiency.npz",
         x=np.log2(xaxis),
         ka=kr.variables[0].samples,
         kb=kr.variables[1].samples,
         ky=kinisi_atoms,
         wa=wr.variables[0].samples,
         wb=wr.variables[1].samples,
         wy=weighted_atoms,
         pa=pr.variables[0].samples,
         pb=pr.variables[1].samples,
         py=ordinary_atoms,
         ta=true[0],
         tb=true[1],
         dta=perr[0],
         dtb=perr[1],
         ty=true_d_atoms.var(-1, ddof=1),
         ka2=kr2.variables[0].samples,
         kb2=kr2.variables[1].samples,
         ky2=kinisi_length,
         wa2=wr2.variables[0].samples,
         wb2=wr2.variables[1].samples,
         wy2=weighted_length,
         pa2=pr2.variables[0].samples,
         pb2=pr2.variables[1].samples,
         py2=ordinary_length,
         ta2=true2[0],
         tb2=true2[1],
         dta2=perr2[0],
         dtb2=perr2[1],
         ty2=true_d_length.var(-1, ddof=1))
