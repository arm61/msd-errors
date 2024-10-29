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
pyblock = np.load(paths.data / f"random_walks/pyblock/rw_{jump}_{atoms}_{length}_s4096.npz")['data']
numerical = np.load(paths.data / f"random_walks/numerical/rw_{jump}_{atoms}_{length}_s4096.npz")['data']
dt = kinisi[0, :, 0]

figsize = (3.73*2, 0.85*3)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.7, hspace=1.4)

axes = []
titles = []

axes.append(fig.add_subplot(gs[0, 0]))
titles.append(r"a")
axes[-1].set_title("numerical")
m, c, b = axes[-1].errorbar(dt, numerical.mean(0), numerical.std(0) * 2, color='#81A8D9', elinewidth=0.2)
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
axes[-1].set_title("kinisi estimated")
m, c, b = axes[-1].errorbar(dt, kinisi[0, :, 1], kinisi[0, :, 2] * 2, color='#E08D6D', elinewidth=0.2)
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
axes[-1].set_title("kinisi $\sigma^2$ comparison")
m = np.max([numerical.var(0), (kinisi[0, :, 2]**2)])
axes[-1].plot(numerical.var(0) / 1000, (kinisi[0, :, 2]**2) / 1000,
              color='#67B99B',
              marker='.',
              ls='',
              markeredgewidth=0,
              markersize=4)
axes[-1].plot(np.linspace(0, m, 10) / 1000, np.linspace(0, m, 10) / 1000, color='#C4C4C4', lw=0.5)
axes[-1].set_xlabel(r"$\sigma^2[x(t)]$ / $10^3$")
axes[-1].set_ylabel(r"$\hat{\sigma}^2[x(t)]$ / $10^3$")
axes[-1].set_xlim(0, None)
axes[-1].set_ylim(0, None)

axes.append(fig.add_subplot(gs[1, 1], sharey=axes[-1]))
titles.append(r"d")
axes[-1].set_title("pyblock estimated")
m, c, b = axes[-1].errorbar(dt, pyblock[0, :, 1], pyblock[0, :, 2] * 2, color='#E08D6D', elinewidth=0.2)
[bar.set_alpha(0.5) for bar in b]
[cap.set_alpha(0.5) for cap in c]
axes[-1].set_xlabel(r"$t^{}$")
axes[-1].set_ylabel(r"$x(t)$")
axes[-1].set_xlim(0, None)
axes[-1].set_ylim(0, None)
axes[-1].set_xticks([0, 64, 128])
axes[-1].set_yticks([0, 400, 800])

axes.append(fig.add_subplot(gs[1, 2]))
titles.append(r"e")
axes[-1].set_title("pyblock $\sigma^2$ comparison")
m = np.max([numerical.var(0), (pyblock[0, :, 2]**2)])
axes[-1].plot(numerical.var(0) / 1000, (pyblock[0, :, 2]**2) / 1000,
              color='#67B99B',
              marker='.',
              ls='',
              markeredgewidth=0,
              markersize=4)
axes[-1].plot(np.linspace(0, m, 10) / 1000, np.linspace(0, m, 10) / 1000, color='#C4C4C4', lw=0.5)
axes[-1].set_xlabel(r"$\sigma^2[x(t)]$ / $10^3$")
axes[-1].set_ylabel(r"$\hat{\sigma}^2[x(t)]$ / $10^3$")
axes[-1].set_xlim(0, None)
axes[-1].set_ylim(0, None)


# axes[-1].set_xticks([0, 40, 80])
# axes[-1].set_yticks([0, 40, 80])

fig.align_ylabels(axes)

x_correction = [33, 33, 27] * 2
for i, ax in enumerate(axes):
    print(ax.get_window_extent().x1 - ax.get_window_extent().x0)
    print(ax.get_window_extent().y1 - ax.get_window_extent().y0)
    x = ax.get_window_extent().x0 - x_correction[i]
    y = ax.get_window_extent().y1 + 11
    x, y = fig.transFigure.inverted().transform([x, y])
    fig.text(x, y, titles[i], ha='left')

fig.set_size_inches(*figsize)
plt.savefig(paths.figures / "msd_blocking.pdf", bbox_inches="tight")
plt.close()
