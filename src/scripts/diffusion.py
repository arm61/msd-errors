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
length = 2000
ll = len([i + length for i in range(0, 20001 - length, length)])
# length = 500
# ll = len([i + length for i in range(0, 2000, length)])
d = np.zeros((6, 8, ll, 3200))
for i in range(0, 6, 1):
    d[i] = np.load(paths.data / f'llzo/diffusion_{i}_10.npz')['d']
d = d.reshape(-1, 3200)
llzo_x = np.linspace(dllzo_true.min(), dllzo_true.max(), 1000)

figsize = (3.2, 1.42)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=1)

axes = []
titles = []

print(d.shape)

axes.append(fig.add_subplot(gs[0, 0]))
y, x = np.histogram(d.mean(-1), bins=int(fp.NBINS * 0.5), density=True)
axes[-1].stairs(y, x, fill=True, color='#ECBCA7', label='$p(\hat{D}^*)$')
axes[-1].plot(llzo_x,
              norm.pdf(llzo_x, dllzo_true.mean(), dllzo_true.std()),
              color='#B8B8B8',
              label='$p(\hat{D}^*_{\mathrm{num}})$')
axes[-1].plot(np.array([d.mean(-1).mean() - d.mean(-1).std(),
                        d.mean(-1).mean() + d.mean(-1).std()]),
              [axes[-1].get_ylim()[1] * 1.05, axes[-1].get_ylim()[1] * 1.05],
              color='#F3ADBC',
              label='$\sigma[\hat{D}^*]$',
              marker="|")
# axes[-1].set_yticks([0, 6, 12])
axes[-1].set_xlabel(r"$\hat{D}^*$ / cm$^2$ s$^{-1}$")
axes[-1].set_ylabel(r"$p(\hat{D}^*)$ / cm$^{-2}$ s")
titles.append("a")
# axes[-1].set_yticks([0, 7, 14])
# axes[-1].set_ylim([0, 16.5])
axes[-1].legend(loc='upper left', bbox_to_anchor=(0.6, 1))

axes.append(fig.add_subplot(gs[0, 1]))
y, x = np.histogram(d.var(-1, ddof=1), bins=int(fp.NBINS * 0.5), density=True)
axes[-1].stairs(y, x, fill=True, color='#B3CBE8', label=r'$p(\hat{\sigma}^2[D^*])$')
axes[-1].axvline(d.mean(-1).var(ddof=1), c='#F3ADBC', ls='-', label=r'$\sigma^2(\hat{D}^*)$')
axes[-1].set_xlabel(r"$\hat{\sigma}^2(\hat{D}^*)$ / cm$^4$ s$^{-2}$")
axes[-1].set_ylabel(r"$p(\hat{\sigma}^2[\hat{D}^*])$ / cm$^{-4}$ s$^2$")
axes[-1].set_xlim([0, None])
# axes[-1].set_xticks([0, 1, 2])
# axes[-1].set_yticks([0, 7, 14])
titles.append(r"b")
axes[-1].legend(loc='upper left', bbox_to_anchor=(0.5, 1))

x_correction = [28, 28]
for i, ax in enumerate(axes):
    x = ax.get_window_extent().x0 - x_correction[i]
    y = ax.get_window_extent().y1 + 11.5
    x, y = fig.transFigure.inverted().transform([x, y])
    fig.text(x, y, titles[i], ha='left')

plt.savefig(paths.figures / "diffusion.pdf", bbox_inches="tight")
plt.close()
