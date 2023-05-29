import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

import utils._fig_params as fp
from utils.plotting_helper import mid_points
import paths

jump = 1
atoms = 128
length = 128

kinisi = np.load(paths.data / f"random_walks/kinisi/rw_{jump}_{atoms}_{length}_s4096.npz")['data']
numerical = np.load(paths.data / f"random_walks/numerical/rw_{jump}_{atoms}_{length}_s4096.npz")['data']
dt = kinisi[0, :, 0]

figsize = (3.73, 0.85)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.7, hspace=0.7)

axes = []
titles = []

axes.append(fig.add_subplot(gs[0, 0]))
titles.append(r"a")
axes[-1].set_title("Numerical")
m, c, b = axes[-1].errorbar(dt, numerical.mean(0), numerical.std(0), color=fp.colors[0], elinewidth=0.2)
[bar.set_alpha(0.5) for bar in b]
[cap.set_alpha(0.5) for cap in c]
axes[-1].set_xlabel(r"$t^{}$")
axes[-1].set_ylabel(r"$x(t)$")
axes[-1].set_xlim(0, None)
axes[-1].set_ylim(0, None)
axes[-1].set_xticks([0, 64, 128])
axes[-1].set_yticks([0, 400, 800])

axes.append(fig.add_subplot(gs[0, 1], sharey=axes[-1]))
titles.append(r"b")
axes[-1].set_title("Estimated")
m, c, b = axes[-1].errorbar(dt, kinisi[0, :, 1], kinisi[0, :, 2], color=fp.colors[2], elinewidth=0.2)
[bar.set_alpha(0.5) for bar in b]
[cap.set_alpha(0.5) for cap in c]
axes[-1].set_xlabel(r"$t^{}$")
axes[-1].set_ylabel(r"$x(t)$")
axes[-1].set_xlim(0, None)
axes[-1].set_ylim(0, None)
axes[-1].set_xticks([0, 64, 128])
axes[-1].set_yticks([0, 400, 800])

axes.append(fig.add_subplot(gs[0, 2]))
titles.append(r"c")
axes[-1].set_title("$\sigma^2$ accuracy")
m = np.max([numerical.var(0), (kinisi[0, :, 2]**2)])
axes[-1].plot(numerical.var(0) / 1000, (kinisi[0, :, 2]**2) / 1000,
              color=fp.colors[1],
              marker='.',
              ls='',
              markeredgewidth=0,
              markersize=4)
axes[-1].plot(np.linspace(0, m, 10) / 1000, np.linspace(0, m, 10) / 1000, color=fp.NEARLY_BLACK, lw=0.5)
axes[-1].set_xlabel(r"$\sigma^2[x(t)]$ / $10^3$")
axes[-1].set_ylabel(r"$\hat{\sigma}^2[x(t)]$ / $10^3$")
axes[-1].set_xlim(0, None)
axes[-1].set_ylim(0, None)
# axes[-1].set_xticks([0, 40, 80])
# axes[-1].set_yticks([0, 40, 80])

fig.align_ylabels(axes)

x_correction = [33, 33, 27]
for i, ax in enumerate(axes):
    print(ax.get_window_extent().x1 - ax.get_window_extent().x0)
    print(ax.get_window_extent().y1 - ax.get_window_extent().y0)
    x = ax.get_window_extent().x0 - x_correction[i]
    y = ax.get_window_extent().y1 + 11
    x, y = fig.transFigure.inverted().transform([x, y])
    fig.text(x, y, titles[i], ha='left')

fig.set_size_inches(*figsize)
plt.savefig(paths.figures / "msd.pdf", bbox_inches="tight")
plt.close()
