import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm, skew
from uravu.distribution import Distribution
from uravu.axis import Axis

import utils._fig_params as fp
from utils.plotting_helper import to_string, mid_points
import paths

jump = 1

xaxis = np.array([16, 32, 64, 128, 256, 512, 1024])

figsize = (3.64, 4.5)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.6, hspace=0.4)
axes = []
titles = []
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']

data = np.load(paths.data / f"random_walks/efficiency.npz")

axes.append(fig.add_subplot(gs[0, 0]))
axes[-1].plot(xaxis, data['ty'], marker='.', ls="", color=fp.colors[0])
a = Axis([Distribution(i.var(-1, ddof=1)) for i in data['ky']])
n = [np.mean(i.var(-1, ddof=1)) for i in data['ky']]
axes[-1].errorbar(xaxis, n, a.ci(), marker='.', ls="", color=fp.colors[2])
axes[-1].set_xscale('log', base=2)
axes[-1].set_yscale('log', base=2)
axes[-1].set_xlabel(r'$N_{{\mathrm{{atoms}}}}$ / atoms')
axes[-1].set_ylabel(r"$\hat{\sigma}^2 (\hat{D}^*)$")
axes[-1].set_xticks([32, 128, 512])
axes[-1].set_xticklabels([32, 128, 512])
axes[-1].minorticks_off()
titles.append(f'{letters[0]}')
axes[-1].set_title(r"$\hat{\sigma}^2(\hat{D}^*)$ scaling")

axes.append(fig.add_subplot(gs[1, 0]))
axes[-1].plot(xaxis, (a.ci()[1] - a.ci()[0]) * 1e5, '.', c=fp.colors[2])
axes[-1].set_xscale('log', base=2)
axes[-1].set_yscale('log', base=2)
axes[-1].set_xlabel(r'$N_{{\mathrm{{atoms}}}}$ / atoms')
axes[-1].set_ylabel(r'Diff. CI bounds / $10^{-5}$')
axes[-1].set_xticks([32, 128, 512])
axes[-1].set_xticklabels([32, 128, 512])
axes[-1].minorticks_off()
titles.append(rf'{letters[2]}')
axes[-1].set_title(r"$p[\hat{\sigma}^2(\hat{D}^*)]$ width scaling")

axes.append(fig.add_subplot(gs[0, 1], sharey=axes[-2]))
axes[-1].plot(xaxis, data['ty2'], marker='.', ls="", color=fp.colors[0])
a = Axis([Distribution(i.var(-1, ddof=1)) for i in data['ky2']])
n = [np.mean(i.var(-1, ddof=1)) for i in data['ky2']]
axes[-1].errorbar(xaxis, n, a.ci(), marker='.', ls="", color=fp.colors[2])
axes[-1].set_xscale('log', base=2)
axes[-1].set_yscale('log', base=2)
axes[-1].set_xlabel(r'$\Delta t_{\mathrm{max}}$')
axes[-1].set_ylabel(r"$\hat{\sigma}^2 (\hat{D}^*)$")
axes[-1].set_xticks([32, 128, 512])
axes[-1].set_xticklabels([32, 128, 512])
axes[-1].minorticks_off()
titles.append(rf'{letters[1]}')
axes[-1].set_title("$\hat{\sigma}^2(\hat{D}^*)$ scaling")

axes.append(fig.add_subplot(gs[1, 1], sharey=axes[-2]))
axes[-1].plot(xaxis, (a.ci()[1] - a.ci()[0]) * 1e5, '.', c=fp.colors[2])
axes[-1].set_xscale('log', base=2)
axes[-1].set_yscale('log', base=2)
axes[-1].set_xlabel(r'$\Delta t_{\mathrm{max}}$')
axes[-1].set_ylabel(r'Diff. CI bounds / $10^{-5}$')
axes[-1].set_xticks([32, 128, 512])
axes[-1].set_xticklabels([32, 128, 512])
axes[-1].minorticks_off()
titles.append(rf'{letters[3]}')
axes[-1].set_title("$p[\hat{\sigma}^2(\hat{D}^*)]$ width scaling")

x, _ = mid_points(axes[0])
y = fig.axes[0].get_window_extent().y1 + 30
x, y = fig.transFigure.inverted().transform([x, y])
fig.text(x, y, 'Varying $N_{\mathrm{atoms}}$', ha='center', fontweight='bold')
x, _ = mid_points(axes[2])
y = fig.axes[0].get_window_extent().y1 + 30
x, y = fig.transFigure.inverted().transform([x, y])
fig.text(x, y, 'Varying $t_{\mathrm{max}}$', ha='center', fontweight='bold')

x_correction = [36, 36, 38.5, 38.5]
for i, ax in enumerate(axes):
    x = ax.get_window_extent().x0 - x_correction[i]
    y = ax.get_window_extent().y1 + 11.7
    x, y = fig.transFigure.inverted().transform([x, y])
    f = fig.text(x, y, titles[i], ha='left')

fig.align_ylabels(axes)

axes[-1].errorbar([], [], [], marker='.', ls='', color=fp.colors[2], label='Model covariance')
axes[-1].plot([], [], marker='.', ls='', color=fp.colors[0], label='Numerical')

plt.figlegend(loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)

plt.savefig(paths.figures / "efficiency.pdf", bbox_inches="tight")
plt.close()