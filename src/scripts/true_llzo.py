import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
from uravu.distribution import Distribution
from uravu.axis import Axis

import utils._fig_params as fp
from utils.plotting_helper import mid_points
import paths

jump = 1
atoms = 128
length = 128
correlation = 'true'
type = 'kinisi'

start_diffs = [0, 2, 4, 6, 8, 10, 15, 20]
d = np.zeros((len(start_diffs), 16, 8, 4, 3200))
timestep = np.load(paths.data / f"llzo/diffusion_0_10.npz")['dt'][0, 0]
dinfty_true = np.zeros((len(start_diffs), 3200))
cinfty_true = np.zeros((len(start_diffs), 3200))
for j, start in enumerate(start_diffs):
    dinfty_true[j] = np.load(paths.data / f"llzo/true_{start}.npz")['diff_c']
    cinfty_true[j] = np.load(paths.data / f"llzo/true_{start}.npz")['intercept']
    for i in range(0, 16, 1):
        d[j, i] = np.load(paths.data / f'llzo/diffusion_{i}_{start}.npz')['d']
d = d.reshape(len(start_diffs), -1, 3200)

figsize = (8.455, 3)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.6, hspace=1)
letters = [['a', 'b', 'c', 'd'], ['e', 'f', 'g', 'h']]
times = [[0, 2, 4, 6], [8, 10, 15, 20]]

axes = []
titles = []

j = 0
for i in [0, 4]:

    axes.append(fig.add_subplot(gs[j, 0]))
    y, x = np.histogram(d[i].var(-1, ddof=1), density=True, bins=int(fp.NBINS * 0.5))
    axes[-1].stairs(y, x, fill=True, color=fp.colors[0])
    axes[-1].axvline(d[i].mean(-1).var(ddof=1), c=fp.colors[3], ls='--')
    axes[-1].set_xlabel(r"$\hat{\sigma}^2 (\hat{D}^*_{\mathrm{Li}^{+}})$ / cm$^4$s$^{-2}$")
    axes[-1].set_ylabel(r"$p[\hat{\sigma}^2 (\hat{D}^*_{\mathrm{Li}^{+}})]$ / cm$^{-4}$s$^2$")
    axes[-1].set_xlim([0, None])
    axes[-1].get_xaxis().get_offset_text().set_position((1.15, 0.1))
    titles.append(f'{letters[j][0]}')
    axes[-1].set_title(r"$t_{\mathrm{diff}} = " + f"{times[j][0]}$ ps")

    axes.append(fig.add_subplot(gs[j, 1]))
    y, x = np.histogram(d[i + 1].var(-1, ddof=1), density=True, bins=int(fp.NBINS * 0.5))
    axes[-1].stairs(y, x, fill=True, color=fp.colors[0])
    axes[-1].axvline(d[i + 1].mean(-1).var(ddof=1), c=fp.colors[3], ls='--')
    axes[-1].set_xlabel(r"$\hat{\sigma}^2 (\hat{D}^*_{\mathrm{Li}^{+}})$ / cm$^4$s$^{-2}$")
    axes[-1].set_ylabel(r"$p[\hat{\sigma}^2 (\hat{D}^*_{\mathrm{Li}^{+}})]$ / cm$^{-4}$s$^2$")
    axes[-1].set_xlim([0, None])
    axes[-1].get_xaxis().get_offset_text().set_position((1.15, 0.1))
    titles.append(f'{letters[j][1]}')
    axes[-1].set_title(r"$t_{\mathrm{diff}} = " + f"{times[j][1]}$ ps")

    axes.append(fig.add_subplot(gs[j, 2]))
    y, x = np.histogram(d[i + 2].var(-1, ddof=1), density=True, bins=int(fp.NBINS * 0.5))
    axes[-1].stairs(y, x, fill=True, color=fp.colors[0])
    axes[-1].axvline(d[i + 2].mean(-1).var(ddof=1), c=fp.colors[3], ls='--')
    axes[-1].set_xlabel(r"$\hat{\sigma}^2 (\hat{D}^*_{\mathrm{Li}^{+}})$ / cm$^4$s$^{-2}$")
    axes[-1].set_ylabel(r"$p[\hat{\sigma}^2 (\hat{D}^*_{\mathrm{Li}^{+}})]$ / cm$^{-4}$s$^2$")
    axes[-1].set_xlim([0, None])
    axes[-1].get_xaxis().get_offset_text().set_position((1.15, 0.1))
    titles.append(f'{letters[j][2]}')
    axes[-1].set_title("$t_{\mathrm{diff}} = " + f"{times[j][2]}$ ps")

    axes.append(fig.add_subplot(gs[j, 3]))
    y, x = np.histogram(d[i + 3].var(-1, ddof=1), density=True, bins=int(fp.NBINS * 0.5))
    axes[-1].stairs(y, x, fill=True, color=fp.colors[0])
    axes[-1].axvline(d[i + 3].mean(-1).var(ddof=1), c=fp.colors[3], ls='--')
    axes[-1].set_xlabel(r"$\hat{\sigma}^2 (\hat{D}^*_{\mathrm{Li}^{+}})$ / cm$^4$s$^{-2}$")
    axes[-1].set_ylabel(r"$p[\hat{\sigma}^2 (\hat{D}^*_{\mathrm{Li}^{+}})]$ / cm$^{-4}$s$^2$")
    axes[-1].set_xlim([0, None])
    titles.append(f'{letters[j][3]}')
    axes[-1].set_title("$t_{\mathrm{diff}} = " + f"{times[j][3]}$ ps")
    axes[-1].get_xaxis().get_offset_text().set_position((1.15, 0.1))
    j += 1

axes[-1].plot([], [],
              marker='s',
              markersize=5,
              ls='',
              color=fp.colors[0],
              markeredgewidth=0,
              label=r'$p[\hat{\sigma}^2(D^*)]$')
axes[-1].plot([], [], marker='', ls='--', color=fp.colors[3], label=r'$\sigma^2(\hat{D}^*)$')

fig.align_ylabels(axes)

x_correction = [22, 31, 22, 22] * 2
for i, ax in enumerate(axes):
    x = ax.get_window_extent().x0 - x_correction[i]
    y = ax.get_window_extent().y1 + 12
    x, y = fig.transFigure.inverted().transform([x, y])
    f = fig.text(x, y, titles[i], ha='left')

plt.figlegend(loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3)

plt.savefig(paths.figures / "true_llzo.pdf", bbox_inches="tight")
plt.close()
