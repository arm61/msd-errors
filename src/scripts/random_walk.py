import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

import utils._fig_params as fp
from utils.plotting_helper import mid_points, CREDIBLE_INTERVALS
import paths

jump = 1
atoms = 128
length = 128
type = 'kinisi'

true = np.load(paths.data / f"random_walks/{type}/rw_{jump}_{atoms}_{length}_s4096.npz")['diff_c']
dinfty_true = np.load(paths.data / f"random_walks/numerical/D_1_128_128.npz")['diff_c']

rw_x = np.linspace(true.mean(-1).min(), true.mean(-1).max(), 1000)

kinisi = np.load(paths.data / f"random_walks/kinisi/rw_{jump}_{atoms}_{length}_s4096.npz")
dt = kinisi['data'][0, :, 0]
distribution = kinisi['diff_c'][0] * 6 * dt[:, np.newaxis] + kinisi['intercept'][0]

figsize = (8.06, 1.3)
print(figsize)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.8, width_ratios=[87, 63, 63, 63])

axes = []
titles = []

axes.append(fig.add_subplot(gs[0, 0]))
titles.append("a")
axes[-1].set_title("Single simulation")
axes[-1].fill_between(dt,
                      *np.percentile(distribution, CREDIBLE_INTERVALS[0][:-1], axis=1),
                      alpha=CREDIBLE_INTERVALS[0][-1],
                      color=fp.colors[1],
                      lw=0,
                      label=r'$p(m | \mathbfit{x})$')
for ci in CREDIBLE_INTERVALS[1:]:
    axes[-1].fill_between(dt, *np.percentile(distribution, ci[:-1], axis=1), alpha=ci[-1], color=fp.colors[1], lw=0)
axes[-1].plot(dt, kinisi['data'][0, :, 1], color='#1E5B84', label=r"$\mathbfit{x}$")
axes[-1].set_xlabel(r"$t$")
axes[-1].set_ylabel(r"$x(t)$")
axes[-1].set_xlim(0, None)
axes[-1].set_ylim(0, None)
axes[-1].set_xticks([0, 64, 128])
axes[-1].set_yticks([0, 400, 800])
axes[-1].legend()

axes.append(fig.add_subplot(gs[0, 1]))
titles.append("b")
axes[-1].set_title("Single simulation")
y, x = np.histogram(kinisi['diff_c'][0], bins=fp.NBINS, density=True)
axes[-1].stairs(y, x, fill=True, color=fp.colors[1], label=r'$p(D^* | \mathbfit{x})$')
axes[-1].plot([
    kinisi['diff_c'][0].mean() - kinisi['diff_c'][0].std(),
    kinisi['diff_c'][0].mean() + kinisi['diff_c'][0].std()
], [axes[-1].get_ylim()[1] * 1.1, axes[-1].get_ylim()[1] * 1.1],
              color=fp.colors[0],
              label='$\hat{\sigma}(D^*)$',
              marker='|')
axes[-1].axvline(kinisi['diff_c'][0].mean(), ymax=0.92, color=fp.colors[2], label=r'$D^*$', ls='--')
axes[-1].set_xlabel(r"$[D^* | \mathbfit{x}]$")
axes[-1].set_ylabel(r"$p\{[D^* | \mathbfit{x}]\}$")
axes[-1].set_xlim(0.87, 1.13)
axes[-1].set_yticks([0, 8, 16])
axes[-1].legend(loc='upper left', bbox_to_anchor=(0.65, 1))

axes.append(fig.add_subplot(gs[0, 2]))
y, x = np.histogram(true.mean(-1), bins=fp.NBINS, density=True)
axes[-1].stairs(y, x, fill=True, color=fp.colors[2], label='$p(\hat{D}^*_{\mathrm{norm}})$')
axes[-1].plot(rw_x,
              norm.pdf(rw_x, dinfty_true.mean(), dinfty_true.std()),
              color=fp.colors[4],
              label='$p(\hat{D}^*_{\mathrm{norm}} | \mathbfit{x})$')
axes[-1].plot([true.mean(-1).mean() - true.mean(-1).std(),
               true.mean(-1).mean() + true.mean(-1).std()],
              [axes[-1].get_ylim()[1] * 1.1, axes[-1].get_ylim()[1] * 1.1],
              color=fp.colors[3],
              label='$\sigma(\hat{D}^*_{\mathrm{norm}})$',
              marker='|')
# axes[-1].set_yticks([0, 4, 8])
axes[-1].set_xlabel(r"$\hat{D}^*_{\mathrm{norm}}$")
axes[-1].set_ylabel(r"$p[\hat{D}^*_{\mathrm{norm}}]$")
titles.append(r"c")
axes[-1].set_title("Many simulations")
axes[-1].set_xlim(0.87, 1.13)
axes[-1].set_yticks([0, 10, 20])
axes[-1].legend(loc='upper left', bbox_to_anchor=(0.6, 1))

axes.append(fig.add_subplot(gs[0, 3]))
y, x = np.histogram(true.var(-1, ddof=1), bins=fp.NBINS, density=True)
axes[-1].stairs(y * 1e-3, x * 1e3, fill=True, color=fp.colors[0], label='$p[\hat{\sigma}^2(D^*)]$')
axes[-1].axvline(true.mean(-1).var(ddof=1) * 1e3,
                 ymax=0.92,
                 c=fp.colors[3],
                 label='$\sigma^2(\hat{D}^*_{\mathrm{norm}})$',
                 ls='--')
axes[-1].set_xlabel(r"$\hat{\sigma}^2 (\hat{D}^*_{\mathrm{norm}})$ / $10^{-3}$")
axes[-1].set_ylabel(r"$p[\hat{\sigma}^2 (\hat{D}^*_{\mathrm{norm}})]$ / $10^3$")
axes[-1].set_xlim([0, None])
# axes[-1].set_xticks([0, 2, 4])
axes[-1].set_yticks([0, 1, 2])
titles.append("d")
axes[-1].set_title("Many simulations")
axes[-1].legend(loc='upper left', bbox_to_anchor=(0.5, 1))

fig.align_ylabels(axes)

x_correction = [33, 28, 27.5, 22.5]
for i, ax in enumerate(axes):
    x = ax.get_window_extent().x0 - x_correction[i]
    y = ax.get_window_extent().y1 + 11.5
    x, y = fig.transFigure.inverted().transform([x, y])
    fig.text(x, y, titles[i], ha='left')

fig.set_size_inches(*figsize)
plt.savefig(paths.figures / "random_walk.pdf", bbox_inches="tight")
plt.close()
