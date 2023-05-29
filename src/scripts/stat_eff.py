import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from uravu.distribution import Distribution
from uravu.axis import Axis

import utils._fig_params as fp
from utils.plotting_helper import mid_points
import paths

correlation = 'true'
jump = 1


def find_exp(number) -> int:
    base10 = np.log10(abs(number))
    return int(abs(np.floor(base10)))


def f(x, a, b):
    return b * x**a


xaxis = np.array([16, 32, 64, 128, 256, 512, 1024])

figsize = (3.66, 2)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.6, hspace=0.4)

x = np.linspace(xaxis.min(), xaxis.max(), 1000)

axes = []
titles = []
letters = ['a', 'b']

data = np.load(paths.data / f"random_walks/stat_eff.npz")

write_out = open(paths.output / f"stat_eff_table.txt", 'w')

write_out.write(r"\multirow{2}{*}{" + r"$1/N_{\mathrm{atoms}}$" + r"} ")
write_out.write(r"& $a$ & \num{" + f"{data['na'] * 10 ** find_exp(data['na']):.2f}" + " \pm " +
                f"{data['dna'] * 10 ** find_exp(data['na']):.2f}e-{find_exp(data['na'])}" + r"} ")
write_out.write(r"& \num{" + f"{data['ka'] * 10 ** find_exp(data['ka']):.2f}" + " \pm " +
                f"{data['dka'] * 10 ** find_exp(data['ka']):.2f}e-{find_exp(data['ka'])}" + r"} ")
write_out.write(r"& \num{" + f"{data['wa'] * 10 ** find_exp(data['wa']):.2f}" + " \pm " +
                f"{data['dwa'] * 10 ** find_exp(data['wa']):.2f}e-{find_exp(data['wa'])}" + r"} ")
write_out.write(r"& \num{" + f"{data['pa'] * 10 ** find_exp(data['pa']):.2f}" + " \pm " +
                f"{data['dpa'] * 10 ** find_exp(data['pa']):.2f}e-{find_exp(data['pa'])}" + r"} \\ ")
write_out.write(r" & $b$/atoms$^{-a}$ & \num{" + f"{data['nb'] * 10 ** find_exp(data['nb']):.2f}" + " \pm " +
                f"{data['dnb'] * 10 ** find_exp(data['nb']):.2f}e-{find_exp(data['nb'])}" + r"} ")
write_out.write(r"& \num{" + f"{data['kb'] * 10 ** find_exp(data['kb']):.2f}" + " \pm " +
                f"{data['dkb'] * 10 ** find_exp(data['kb']):.2f}e-{find_exp(data['kb'])}" + r"} ")
write_out.write(r"& \num{" + f"{data['wb'] * 10 ** find_exp(data['wb']):.2f}" + " \pm " +
                f"{data['dwb'] * 10 ** find_exp(data['wb']):.2f}e-{find_exp(data['wb'])}" + r"} ")
write_out.write(r"& \num{" + f"{data['pb'] * 10 ** find_exp(data['pb']):.2f}" + " \pm " +
                f"{data['dpb'] * 10 ** find_exp(data['pb']):.2f}e-{find_exp(data['pb'])}" + r"} \\ ")

write_out.write(r"\multirow{2}{*}{" + r"$1/\Delta t_{\max}$" + r"} ")
write_out.write(r"& $a$ & \num{" + f"{data['na2'] * 10 ** find_exp(data['na2']):.2f}" + " \pm " +
                f"{data['dna2'] * 10 ** find_exp(data['na2']):.2f}e-{find_exp(data['na2'])}" + r"} ")
write_out.write(r"& \num{" + f"{data['ka2'] * 10 ** find_exp(data['ka2']):.2f}" + " \pm " +
                f"{data['dka2'] * 10 ** find_exp(data['ka2']):.2f}e-{find_exp(data['ka2'])}" + r"} ")
write_out.write(r"& \num{" + f"{data['wa2'] * 10 ** find_exp(data['wa2']):.2f}" + " \pm " +
                f"{data['dwa2'] * 10 ** find_exp(data['wa2']):.2f}e-{find_exp(data['wa2'])}" + r"} ")
write_out.write(r"& \num{" + f"{data['pa2'] * 10 ** find_exp(data['pa2']):.2f}" + " \pm " +
                f"{data['dpa2'] * 10 ** find_exp(data['pa2']):.2f}e-{find_exp(data['pa2'])}" + r"} \\ ")
write_out.write(r" & $b$ & \num{" + f"{data['nb2'] * 10 ** find_exp(data['nb2']):.2f}" + " \pm " +
                f"{data['dnb2'] * 10 ** find_exp(data['nb2']):.2f}e-{find_exp(data['nb2'])}" + r"} ")
write_out.write(r"& \num{" + f"{data['kb2'] * 10 ** find_exp(data['kb2']):.2f}" + " \pm " +
                f"{data['dkb2'] * 10 ** find_exp(data['kb2']):.2f}e-{find_exp(data['kb2'])}" + r"} ")
write_out.write(r"& \num{" + f"{data['wb2'] * 10 ** find_exp(data['wb2']):.2f}" + " \pm " +
                f"{data['dwb2'] * 10 ** find_exp(data['wb2']):.2f}e-{find_exp(data['wb2'])}" + r"} ")
write_out.write(r"& \num{" + f"{data['pb2'] * 10 ** find_exp(data['pb2']):.2f}" + " \pm " +
                f"{data['dpb2'] * 10 ** find_exp(data['pb2']):.2f}e-{find_exp(data['pb2'])}" + r"} \\ ")

write_out.close()

axes.append(fig.add_subplot(gs[0, 0]))
axes[-1].plot(xaxis, data['ky'], marker='.', ls="", color=fp.colors[2])
k = f(x, data['ka'], data['kb'])
axes[-1].plot(x, k, color=fp.colors[2])
axes[-1].plot(xaxis, data['wy'], marker='.', ls="", color=fp.colors[1])
k = f(x, data['wa'], data['wb'])
axes[-1].plot(x, k, color=fp.colors[1])
axes[-1].plot(xaxis, data['py'], marker='.', ls="", color=fp.colors[3])
k = f(x, data['pa'], data['pb'])
axes[-1].plot(x, k, color=fp.colors[3])
axes[-1].plot(xaxis, data['ny'], marker='.', ls="", color=fp.colors[0])
k = f(x, data['na'], data['nb'])
axes[-1].plot(x, k, color=fp.colors[0])
axes[-1].set_xscale('log', base=2)
axes[-1].set_yscale('log', base=2)
axes[-1].set_xlabel(r'$N_{{\mathrm{{atoms}}}}$ / atoms')
axes[-1].set_ylabel(r"$\sigma^2 (\langle\hat{D}^*\rangle)$")
titles.append(f"{letters[0]}")
axes[-1].set_title(r"Varying $N_{\mathrm{atoms}}$")
axes[-1].set_xticks([32, 128, 512])
axes[-1].set_xticklabels([32, 128, 512])
axes[-1].minorticks_off()

axes.append(fig.add_subplot(gs[0, 1]))
axes[-1].plot(xaxis, data['ky2'], marker='.', ls="", color=fp.colors[2])
k = f(x, data['ka2'], data['kb2'])
axes[-1].plot(x, k, color=fp.colors[2])
axes[-1].plot(xaxis, data['wy2'], marker='.', ls="", color=fp.colors[1])
k = f(x, data['wa2'], data['wb2'])
axes[-1].plot(x, k, color=fp.colors[1])
axes[-1].plot(xaxis, data['py2'], marker='.', ls="", color=fp.colors[3])
k = f(x, data['pa2'], data['pb2'])
axes[-1].plot(x, k, color=fp.colors[3])
axes[-1].plot(xaxis, data['ny2'], marker='.', ls="", color=fp.colors[0])
k = f(x, data['na2'], data['nb2'])
axes[-1].plot(x, k, color=fp.colors[0])
axes[-1].set_xscale('log', base=2)
axes[-1].set_yscale('log', base=2)
axes[-1].set_xlabel(r'$\Delta t_{\mathrm{max}}$')
axes[-1].set_ylabel(r"$\sigma^2 (\langle\hat{D}^*\rangle)$")
titles.append(f"{letters[1]}")
axes[-1].set_title(r"Varying $t_{\max}$")
axes[-1].set_xticks([32, 128, 512])
axes[-1].set_xticklabels([32, 128, 512])
axes[-1].minorticks_off()

axes[-1].plot([], [], marker='.', ls='-', color=fp.colors[0], label='Numerical')
axes[-1].plot([], [], marker='.', ls='-', color=fp.colors[1], label='Independent')
axes[-1].plot([], [], marker='.', ls='-', color=fp.colors[2], label='Model Covariance')
axes[-1].plot([], [], marker='.', ls='-', color=fp.colors[3], label='Independent & Identical')

plt.figlegend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2)

x_correction = [36] * 2
for i, ax in enumerate(axes):
    x = ax.get_window_extent().x0 - x_correction[i]
    y = ax.get_window_extent().y1 + 11.5
    x, y = fig.transFigure.inverted().transform([x, y])
    f = fig.text(x, y, titles[i], ha='left')

plt.savefig(paths.figures / "stat_eff.pdf", bbox_inches="tight")
plt.close()
