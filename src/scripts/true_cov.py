import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
from uravu.distribution import Distribution
from uravu.axis import Axis

import utils._fig_params as fp
from utils.plotting_helper import to_string, CREDIBLE_INTERVALS
import paths

jump = 1
atoms = 128
length = 128
type = 'true_cov'

true = np.load(paths.data / f"random_walks/{type}/rw_{jump}_{atoms}_{length}_s4096.npz")['diff_c']
dinfty_true = np.load(paths.data / "random_walks/numerical/D_1_128_128.npz")['diff_c']

figsize = (3.4, 1.4)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=1)

axes = []
titles = []

x_ = np.linspace(true.mean(-1).min(), true.mean(-1).max(), 1000)

axes.append(fig.add_subplot(gs[0, 0]))
y, x = np.histogram(true.mean(-1), density=True, bins=fp.NBINS)
axes[-1].stairs(y * 1e-1, x, fill=True, color=fp.colors[2], label='$p(\hat{D}^*)$')
axes[-1].plot(x_,
              norm.pdf(x_, dinfty_true.mean(), dinfty_true.std()) * 1e-1,
              color=fp.colors[4],
              label=r'$p(\hat{D}^* | \mathbfit{x})$')
axes[-1].plot([dinfty_true.mean() - dinfty_true.std(),
               dinfty_true.mean() + dinfty_true.std()], [axes[-1].get_ylim()[1] * 1.05, axes[-1].get_ylim()[1] * 1.05],
              color=fp.colors[3],
              label='$\sigma(\hat{D}^*)$')
axes[-1].set_xlabel(r"$\hat{D}^*$")
axes[-1].set_ylabel(r"$p(\hat{D}^*)$")
titles.append(r"a")
axes[-1].set_title("$D^*$ distribution")
axes[-1].legend(loc='upper left', bbox_to_anchor=(0.65, 1))

axes.append(fig.add_subplot(gs[0, 1]))
y, x = np.histogram(true.var(-1, ddof=1), density=True, bins=fp.NBINS)
axes[-1].stairs(y * 1e-5, x * 1e5, fill=True, color=fp.colors[0], label='$p[\hat{\sigma}^2(D^*)]$')
axes[-1].axvline(true.mean(-1).var(ddof=1) * 1e5, c=fp.colors[3], label='$\sigma^2(\hat{D}^*)$', ls='--')
axes[-1].set_xlabel(r"$\hat{\sigma}^2 (\hat{D}^*)$")
axes[-1].set_ylabel(r"$p[\hat{\sigma}^2 (\hat{D}^*)]$")
# axes[-1].set_xlim(1.5, 2.0)
titles.append("b")
axes[-1].set_title(r"$\hat{\sigma}^2(D^*)$ distribution")
axes[-1].legend(loc='upper left', bbox_to_anchor=(0.65, 1))

x_correction = [31, 32]
for i, ax in enumerate(axes):
    x = ax.get_window_extent().x0 - x_correction[i]
    y = ax.get_window_extent().y1 + 12
    x, y = fig.transFigure.inverted().transform([x, y])
    f = fig.text(x, y, titles[i], ha='left')

plt.savefig(paths.figures / "true_cov.pdf", bbox_inches="tight")
plt.close()
