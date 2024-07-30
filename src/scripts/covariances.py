import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns

import utils._fig_params as fp
import paths

colors = ["#0173B2", "#029E73", "#D55E00", "#CC78BC", "#ECE133", "#56B4E9"]

jump = 1
dimensionality = 3
D = 6 / (dimensionality * 2)
length = 128
atoms = 128
label_offset = (0, 1.1)
rng = np.random.RandomState(0)
np.random.seed(0)

newcmp1 = sns.light_palette(colors[0], as_cmap=True)
newcmp2 = sns.light_palette(colors[1], as_cmap=True)
newcmp3 = sns.light_palette(colors[2], as_cmap=True)
newcmp4 = sns.diverging_palette(311.4, 161.2, as_cmap=True)

true_msd = np.load(paths.data / f"random_walks/numerical/rw_{jump}_{atoms}_{length}_s4096.npz")["data"]
true_cov = np.cov(true_msd[:, 1:length].T)

kinisi_data = np.load(paths.data / f"random_walks/kinisi/rw_{jump}_{atoms}_{length}_s512.npz")

kinisi_cov = kinisi_data["covariance"].mean(0)
no = (kinisi_data["n_o"]).mean(0)# * kinisi_data['f'][:, 4, np.newaxis]).mean(0)
timestep = np.arange(1, length + 1, 1)

ts_mesh = np.meshgrid(timestep[1:], timestep[1:])

anal_cov = np.zeros((length, length))
for i in range(0, anal_cov.shape[0]):
    for j in range(i, anal_cov.shape[1]):
        value = 8 * dimensionality**2 * D**2 * timestep[i]**2 / (dimensionality * no[j])
        anal_cov[i, j] = value
        anal_cov[j, i] = np.copy(anal_cov[i, j])
anal_cov = anal_cov[1:, 1:]

max_cov = np.ceil(np.max([anal_cov.max(), true_cov.max(), kinisi_cov.max()]))
min_cov = np.floor(np.min([anal_cov.min(), true_cov.min(), kinisi_cov.min()]))
levels = np.linspace(min_cov, max_cov, 15)

figsize = (5.0, 4)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.8, hspace=0.8)

axes = []
titles = []
titles2 = []

axes.append(fig.add_subplot(gs[0, 0]))
im1 = axes[-1].contourf(ts_mesh[0], ts_mesh[1], true_cov, cmap=newcmp1, levels=levels)
fig.subplots_adjust(right=0.8)
cb1 = fig.colorbar(im1, fraction=0.046)
cb1.ax.tick_params(labelsize=fp.FONTSIZE, width=0.5)
cb1.ax.yaxis.set_ticks([0, 1500, 3000])
cb1.outline.set_linewidth(0.5)
axes[-1].set_xticks([0, int(length / 2), length])
axes[-1].set_yticks([0, int(length / 2), length])
# axes[-1].set_aspect("equal")
titles.append(r"a")
titles2.append("numerical")

axes.append(fig.add_subplot(gs[1, 1]))
im1 = axes[-1].contourf(ts_mesh[0], ts_mesh[1], anal_cov, cmap=newcmp2, levels=levels)
fig.subplots_adjust(right=0.8)
cb1 = fig.colorbar(im1, fraction=0.046)
cb1.ax.tick_params(labelsize=fp.FONTSIZE, width=0.5)
cb1.ax.yaxis.set_ticks([0, 1500, 3000])
cb1.outline.set_linewidth(0.5)
axes[-1].set_xticks([0, int(length / 2), length])
axes[-1].set_yticks([0, int(length / 2), length])
titles.append(r"b")
titles2.append("analytical")

axes.append(fig.add_subplot(gs[2, 2]))
im1 = axes[-1].contourf(ts_mesh[0], ts_mesh[1], kinisi_cov, cmap=newcmp3, levels=levels)
fig.subplots_adjust(right=0.8)
cb1 = fig.colorbar(im1, fraction=0.046)
cb1.ax.tick_params(labelsize=fp.FONTSIZE, width=0.5)
cb1.ax.yaxis.set_ticks([0, 1500, 3000])
cb1.outline.set_linewidth(0.5)
axes[-1].set_xticks([0, int(length / 2), length])
axes[-1].set_yticks([0, int(length / 2), length])
titles.append(r"c")
titles2.append("estimated")

axes.append(fig.add_subplot(gs[0, 1]))
ratio = (true_cov / anal_cov)
mod = 0.5
im1 = axes[-1].contourf(ts_mesh[0], ts_mesh[1], ratio, cmap=newcmp4, levels=np.linspace(1 - mod, 1 + mod, 15))
fig.subplots_adjust(right=0.8)
cb1 = fig.colorbar(im1, fraction=0.046)
cb1.ax.tick_params(labelsize=fp.FONTSIZE, width=0.5)
cb1.ax.yaxis.set_ticks([0.5, 1.0, 1.5])
cb1.outline.set_linewidth(0.5)
axes[-1].set_xticks([0, int(length / 2), length])
axes[-1].set_yticks([0, int(length / 2), length])
titles.append(r"$\mathrm{a}\div\mathrm{b}$")

axes.append(fig.add_subplot(gs[0, 2]))
ratio = (true_cov / kinisi_cov)
im1 = axes[-1].contourf(ts_mesh[0], ts_mesh[1], ratio, cmap=newcmp4, levels=np.linspace(1 - mod, 1 + mod, 15))
fig.subplots_adjust(right=0.8)
cb1 = fig.colorbar(im1, fraction=0.046)
cb1.ax.tick_params(labelsize=fp.FONTSIZE, width=0.5)
cb1.ax.yaxis.set_ticks([0.5, 1.0, 1.5])
cb1.outline.set_linewidth(0.5)
axes[-1].set_xticks([0, int(length / 2), length])
axes[-1].set_yticks([0, int(length / 2), length])
titles.append(r"$\mathrm{a}\div\mathrm{c}$")

axes.append(fig.add_subplot(gs[1, 2]))
im1 = axes[-1].contourf(ts_mesh[0],
                        ts_mesh[1], (anal_cov / kinisi_cov),
                        cmap=newcmp4,
                        levels=np.linspace(1 - mod, 1 + mod, 15))
fig.subplots_adjust(right=0.8)
cb1 = fig.colorbar(im1, fraction=0.046)
cb1.ax.tick_params(labelsize=fp.FONTSIZE, width=0.5)
cb1.ax.yaxis.set_ticks([0.5, 1, 1.5])
cb1.ax.ticklabel_format(style='plain', useOffset=False)
cb1.outline.set_linewidth(0.5)
axes[-1].set_xticks([0, int(length / 2), length])
axes[-1].set_yticks([0, int(length / 2), length])
titles.append(r"$\mathrm{b}\div\mathrm{c}$")

x_correction = [0] * 6
for i, ax in enumerate(axes):
    print(ax.get_window_extent().x1 - ax.get_window_extent().x0)
    print(ax.get_window_extent().y1 - ax.get_window_extent().y0)
    x = ax.get_window_extent().x0 - x_correction[i]
    y = ax.get_window_extent().y1 + 11.2 
    x, y = fig.transFigure.inverted().transform([x, y])
    fig.text(x, y, titles[i], ha='left')
for i, ax in enumerate(axes[:3]):
    ax.text(0.5, 0.1, titles2[i], ha='center', va='bottom', transform=ax.transAxes)

fig.set_size_inches(*figsize)
plt.savefig(paths.figures / "covariances.pdf", bbox_inches="tight")
plt.close()
