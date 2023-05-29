import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

import utils._fig_params as fp
from utils.plotting_helper import CREDIBLE_INTERVALS
import paths

jump = 1
atoms = 128
length = 128

kinisi = np.load(paths.data / f"random_walks/kinisi/rw_{jump}_{atoms}_{length}_s4096.npz")
dt = kinisi['data'][0, :, 0]
distribution = kinisi['diff_c'][0] * 6 * dt[:, np.newaxis] + kinisi['intercept'][0]

true = 1 / 6

figsize = (3.4, 3.4)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.9, hspace=0.7)

axes = []
titles = []

axes.append(fig.add_subplot(gs[0, :3]))
titles.append(r"$\mathbf{a}$ - $x(t)$")
for ci in CREDIBLE_INTERVALS:
    axes[-1].fill_between(dt,
                          *np.percentile(distribution, ci[:-1], axis=1) * 1e-1,
                          alpha=ci[-1],
                          color=fp.colors[2],
                          lw=0)
axes[-1].plot(dt, kinisi['data'][0, :, 1] * 1e-1, color=fp.colors[1], label=r'$x(t)$')
axes[-1].set_xlabel(r"$t$ / s")
axes[-1].set_ylabel(r"$x(t)$ / $10^1$m$^2$")
axes[-1].set_xlim(0, None)
axes[-1].set_ylim(0, None)
axes[-1].set_xticks([0, 64, 128])
axes[-1].set_yticks([0, 6, 12])

axes.append(fig.add_subplot(gs[0, 3:]))
titles.append(r"$\mathbf{b}$ - $p(\hat{D}^* | x(t))$")
y, x = np.histogram(kinisi['diff_c'], bins=fp.NBINS, density=True)
axes[-1].stairs(y * 1e-1, x, fill=True, color=fp.colors[2], label=r'Estimated posterior')
axes[-1].axvline(true, color=fp.colors[0], label='True $D^*$')
axes[-1].set_xlabel(r"$D^*$ / m$^2$s$^{-1}$")
axes[-1].set_ylabel(r"$p[D^* | x(t)]$ / $10^{1}$m$^{-2}$s")
axes[-1].set_xlim(0.145, 0.19)

x_correction = [30.5, 21]
for i, ax in enumerate(axes):
    x = ax.get_window_extent().x0 - x_correction[i]
    y = ax.get_window_extent().y1 + 15
    x, y = fig.transFigure.inverted().transform([x, y])
    fig.text(x, y, titles[i], ha='left')

plt.figlegend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=3)

fig.set_size_inches(*figsize)
plt.savefig(paths.figures / "bmvn_msd.pdf", bbox_inches="tight")
plt.close()
