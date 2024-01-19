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

kinisi = np.load(paths.data / f"random_walks/kinisi/rw_{jump}_{atoms}_{length}_s4096.npz")['diff_c']
pyblock = np.load(paths.data / f"random_walks/pyblock/rw_{jump}_{atoms}_{length}_s4096.npz")['diff_c']
pyblock_mf = np.load(paths.data / f"random_walks/pyblock_modelfree/rw_{jump}_{atoms}_{length}_s4096.npz")['diff_c']
dinfty_true = np.load(paths.data / f"random_walks/numerical/D_1_128_128.npz")['diff_c']

rw_x = np.arange(kinisi.mean(-1).min(), kinisi.mean(-1).max(), 0.01)
rw_y = np.arange(pyblock.mean(-1).min(), pyblock.mean(-1).max(), 0.01)
rw_z = np.arange(pyblock_mf.mean(-1).min(), pyblock_mf.mean(-1).max(), 0.01)
rw_x2 = np.arange(np.array([kinisi.var(-1, ddof=1), pyblock.var(-1, ddof=1), pyblock_mf.var(-1, ddof=1)]).min(), 
                  np.array([kinisi.var(-1, ddof=1), pyblock.var(-1, ddof=1), pyblock_mf.var(-1, ddof=1)]).max(), 0.0001)
rw_x3 = np.arange(kinisi.mean(-1).min(), kinisi.mean(-1).max(), 0.0001)
rw_y3 = np.arange(pyblock.mean(-1).min(), pyblock.mean(-1).max(), 0.0001)
rw_z3 = np.arange(pyblock_mf.mean(-1).min(), pyblock_mf.mean(-1).max(), 0.0001)

figsize = (4.03, 6)
print(figsize)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.6, hspace=1)

axes = []
titles = []

axes.append(fig.add_subplot(gs[0, 0]))
y, x = np.histogram(kinisi.mean(-1), bins=rw_x, density=True)
axes[-1].stairs(y, x, fill=True, color='#ECBCA7', label='$p(\hat{D}^*)$')
axes[-1].plot(rw_x3,
              norm.pdf(rw_x3, dinfty_true.mean(), dinfty_true.std()),
              color='#B8B8B8',
              label='$p(\hat{D}^*_{\mathrm{num}})$')
axes[-1].plot([kinisi.mean(-1).mean() - kinisi.mean(-1).std(),
               kinisi.mean(-1).mean() + kinisi.mean(-1).std()],
              [axes[-1].get_ylim()[1] * 1.1, axes[-1].get_ylim()[1] * 1.1],
              color='#F3ADBC',
              label='$\sigma[\hat{D}^*]$',
              marker='|')
axes[-1].set_yticks([0, 10, 20])
axes[-1].set_xlabel(r"$\hat{D}^*$")
axes[-1].set_ylabel(r"$p(\hat{D}^*)$")
titles.append(r"a")
axes[-1].set_title("kinisi")
axes[-1].set_xlim(rw_x3.min(), rw_x3.max())
axes[-1].set_yticks([0, 10, 20])
axes[-1].legend(loc='upper left', bbox_to_anchor=(0.6, 1))

axes.append(fig.add_subplot(gs[0, 1]))
y, x = np.histogram(kinisi.var(-1, ddof=1), bins=rw_x2, density=True)
axes[-1].stairs(y, x, fill=True, color='#B3CBE8', label='$p(\hat{\sigma}^2[D^*])$')
axes[-1].axvline(kinisi.mean(-1).var(ddof=1),
                 ymax=0.95,
                 c='#F3ADBC',
                 label='$\sigma^2[\hat{D}^*]$',
                 ls='-')
axes[-1].set_xlabel(r"$\sigma^2 [\hat{D}^*]$")
axes[-1].set_ylabel(r"$p(\sigma^2 [\hat{D}^*])$")
axes[-1].set_xlim([0, None])
# axes[-1].set_yticks([0, 1, 2, 3])
titles.append("b")
axes[-1].set_title("kinisi")
axes[-1].legend(loc='upper left', bbox_to_anchor=(0.5, 1))

axes.append(fig.add_subplot(gs[1, 0]))
y, x = np.histogram(pyblock.mean(-1), bins=rw_x, density=True)
axes[-1].stairs(y, x, fill=True, color='#ECBCA7', label='$p(\hat{D}^*)$')
axes[-1].plot(rw_y3,
              norm.pdf(rw_y3, dinfty_true.mean(), dinfty_true.std()),
              color='#B8B8B8',
              label='$p(\hat{D}^*_{\mathrm{num}})$')
axes[-1].plot([pyblock.mean(-1).mean() - pyblock.mean(-1).std(),
               pyblock.mean(-1).mean() + pyblock.mean(-1).std()],
              [axes[-1].get_ylim()[1] * 1.1, axes[-1].get_ylim()[1] * 1.1],
              color='#F3ADBC',
              label='$\sigma[\hat{D}^*]$',
              marker='|')
axes[-1].set_yticks([0, 10, 20])
axes[-1].set_xlabel(r"$\hat{D}^*$")
axes[-1].set_ylabel(r"$p(\hat{D}^*)$")
titles.append(r"c")
axes[-1].set_title("blocking fitted")
axes[-1].set_xlim(rw_y3.min(), rw_y3.max())
axes[-1].set_yticks([0, 10, 20])
axes[-1].legend(loc='upper left', bbox_to_anchor=(0.6, 1))

axes.append(fig.add_subplot(gs[1, 1], sharex=axes[-2]))
y, x = np.histogram(pyblock.var(-1, ddof=1), bins=rw_x2, density=True)
axes[-1].stairs(y, x, fill=True, color='#B3CBE8', label='$p(\hat{\sigma}^2[D^*])$')
axes[-1].axvline(pyblock.mean(-1).var(ddof=1),
                 ymax=0.95,
                 c='#F3ADBC',
                 label='$\sigma^2[\hat{D}^*]$',
                 ls='-')
axes[-1].set_xlabel(r"$\sigma^2 [\hat{D}^*]$")
axes[-1].set_ylabel(r"$p(\sigma^2 [\hat{D}^*])$")
axes[-1].set_xlim([0, None])
titles.append("d")
axes[-1].set_title("blocking fitted")
axes[-1].legend(loc='upper left', bbox_to_anchor=(0.5, 1))


axes.append(fig.add_subplot(gs[2, 0]))
y, x = np.histogram(pyblock_mf.mean(-1), bins=rw_z, density=True)
axes[-1].stairs(y, x, fill=True, color='#ECBCA7', label='$p(\hat{D}^*)$')
axes[-1].plot(rw_z3,
              norm.pdf(rw_z3, dinfty_true.mean(), dinfty_true.std()),
              color='#B8B8B8',
              label='$p(\hat{D}^*_{\mathrm{num}})$')
axes[-1].plot([pyblock_mf.mean(-1).mean() - pyblock_mf.mean(-1).std(),
               pyblock_mf.mean(-1).mean() + pyblock_mf.mean(-1).std()],
              [axes[-1].get_ylim()[1] * 1.1, axes[-1].get_ylim()[1] * 1.1],
              color='#F3ADBC',
              label='$\sigma[\hat{D}^*]$',
              marker='|')
axes[-1].set_yticks([0, 10, 20])
axes[-1].set_xlabel(r"$\hat{D}^*$")
axes[-1].set_ylabel(r"$p(\hat{D}^*)$")
titles.append(r"e")
axes[-1].set_title("blocking")
axes[-1].set_xlim(rw_z3.min(), rw_z3.max())
axes[-1].set_yticks([0, 10, 20])
axes[-1].legend(loc='upper left', bbox_to_anchor=(0.6, 1))

axes.append(fig.add_subplot(gs[2, 1], sharex=axes[-2]))
y, x = np.histogram(pyblock_mf.var(-1, ddof=1), bins=rw_x2, density=True)
axes[-1].stairs(y, x, fill=True, color='#B3CBE8', label='$p(\hat{\sigma}^2[D^*])$')
axes[-1].axvline(pyblock_mf.mean(-1).var(ddof=1),
                 ymax=0.95,
                 c='#F3ADBC',
                 label='$\sigma^2[\hat{D}^*]$',
                 ls='-')
axes[-1].set_xlabel(r"$\sigma^2 [\hat{D}^*]$")
axes[-1].set_ylabel(r"$p(\sigma^2 [\hat{D}^*])$")
axes[-1].set_xlim([0, None])
# axes[-1].set_yticks([0, 1, 2, 3])
titles.append("f")
axes[-1].set_title("blocking")
axes[-1].legend(loc='upper left', bbox_to_anchor=(0.5, 1))


fig.align_ylabels(axes)

x_correction = [33, 28, 33, 28, 33, 28]
for i, ax in enumerate(axes):
    x = ax.get_window_extent().x0 - x_correction[i]
    y = ax.get_window_extent().y1 + 11.5
    x, y = fig.transFigure.inverted().transform([x, y])
    fig.text(x, y, titles[i], ha='left')

fig.set_size_inches(*figsize)
plt.savefig(paths.figures / "pyblock.pdf", bbox_inches="tight")
plt.close()
