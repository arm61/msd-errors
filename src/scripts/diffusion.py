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
type = 'kinisi'

dllzo_true = np.load(paths.data / "llzo/true_10.npz")['diff_c']
d = np.zeros((16, 8, 4, 3200))
for i in range(0, 16, 1):
    d[i] = np.load(paths.data / f'llzo/diffusion_{i}_10.npz')['d']
d = d.reshape(-1, 3200)
llzo_x = np.linspace(dllzo_true.min(), dllzo_true.max(), 1000)

figsize = (3.2, 1.42)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=1)

axes = []
titles = []

axes.append(fig.add_subplot(gs[0, 0]))
y, x = np.histogram(d.mean(-1), bins=int(fp.NBINS * 0.5), density=True)
axes[-1].stairs(y * 1e-4, x * 1e5, fill=True, color=fp.colors[2], label='$p(\hat{D}^*_{\mathrm{Li}^+})$')
axes[-1].plot(llzo_x * 1e5,
              norm.pdf(llzo_x, dllzo_true.mean(), dllzo_true.std()) * 1e-4,
              color=fp.colors[4],
              label='$p(\hat{D}^*_{\mathrm{Li}^+} | \mathbfit{x})$')
axes[-1].plot(np.array([d.mean(-1).mean() - d.mean(-1).std(),
                        d.mean(-1).mean() + d.mean(-1).std()]) * 1e5,
              [axes[-1].get_ylim()[1] * 1.05, axes[-1].get_ylim()[1] * 1.05],
              color=fp.colors[3],
              label='$\sigma(\hat{D}^*_{\mathrm{Li}^+})$',
              marker="|")
axes[-1].set_yticks([0, 6, 12])
axes[-1].set_xlabel(r"$\hat{D}^*_{\mathrm{Li}^{+}}$ / $10^{-5}$ cm$^2$ s$^{-1}$")
axes[-1].set_ylabel(r"$p(\hat{D}^*_{\mathrm{Li}^{+}})$ / $10^4$ cm$^{-2}$ s")
titles.append("a")
axes[-1].set_title("$D^*_{\mathrm{Li}^+}$ estimates", y=1)
axes[-1].set_yticks([0, 7, 14])
axes[-1].set_ylim([0, 16.5])
axes[-1].legend(loc='upper left', bbox_to_anchor=(0.6, 1))

axes.append(fig.add_subplot(gs[0, 1]))
y, x = np.histogram(d.var(-1, ddof=1), bins=int(fp.NBINS * 0.5), density=True)
axes[-1].stairs(y * 1e-10, x * 1e11, fill=True, color=fp.colors[0], label='$p[\hat{\sigma}^2(D^*_{\mathrm{Li}^+})]$')
axes[-1].axvline(d.mean(-1).var(ddof=1) * 1e11, c=fp.colors[3], ls='--', label='$\sigma^2(\hat{D}^*_{\mathrm{Li}^+})$')
axes[-1].set_xlabel(r"$\hat{\sigma}^2(\hat{D}^*_{\mathrm{Li}^{+}})$ / $10^{-11}$ cm$^4$ s$^{-2}$")
axes[-1].set_ylabel(r"$p[\hat{\sigma}^2(\hat{D}^*_{\mathrm{Li}^{+}})]$ / $10^{10}$ cm$^{-4}$ s$^2$")
axes[-1].set_xlim([0, None])
axes[-1].set_xticks([0, 1, 2])
axes[-1].set_yticks([0, 7, 14])
axes[-1].set_ylim([0, 16.5])
titles.append(r"b")
axes[-1].set_title("$\sigma^2(D^*_{\mathrm{Li}^+})$ estimates", y=1)
axes[-1].legend(loc='upper left', bbox_to_anchor=(0.5, 1))

x_correction = [28, 28]
for i, ax in enumerate(axes):
    x = ax.get_window_extent().x0 - x_correction[i]
    y = ax.get_window_extent().y1 + 11.5
    x, y = fig.transFigure.inverted().transform([x, y])
    fig.text(x, y, titles[i], ha='left')

plt.savefig(paths.figures / "diffusion.pdf", bbox_inches="tight")
plt.close()
