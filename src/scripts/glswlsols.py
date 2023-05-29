import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
from scipy.stats import norm
from uravu.distribution import Distribution

import utils._fig_params as fp
import paths

data = np.load('src/data/random_walks/numerical/glswlsols_1_128_128.npz')

figsize = (3.744, 3.4)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.8)

x = np.linspace(0.75, 1.25, 1000)

axes = []
titles = []
axes.append(fig.add_subplot(gs[0, 0]))
bins = np.linspace((data['ols_pop']).min(), (data['ols_pop']).max(), fp.NBINS * 2)
axes[-1].hist(data['ols_pop'], bins=bins, density=True, color=fp.colors[5])
axes[-1].plot(x, norm.pdf(x, *norm.fit(data['ols_pop'])), color=fp.colors[5])
axes[-1].plot([1 - data['ols_pop'].std(), 1 + data['ols_pop'].std()],
              [axes[-1].get_ylim()[1] * 1.1, axes[-1].get_ylim()[1] * 1.1],
              color=fp.colors[3],
              marker='|')
axes[-1].plot([1 - data['ols_est'][:, 0].std(), 1 + data['ols_est'][:, 0].std()],
              [axes[-1].get_ylim()[1] * 1.1, axes[-1].get_ylim()[1] * 1.1],
              color='#1E5B84',
              marker='|')
axes[-1].set_xticks([0.8, 0.9, 1, 1.1, 1.2])
axes[-1].set_xlim(0.75, 1.25)
axes[-1].set_yticks([0, 3, 6])
axes[-1].set_ylabel(r'$p[\hat{D}^* (D^*)^{-1}]$')
titles.append("a")
axes[-1].set_title("OLS")

axes.append(fig.add_subplot(gs[1, 0], sharex=axes[0]))
axes[-1].hist(data['wls_pop'], bins=bins, density=True, color=fp.colors[5])
axes[-1].plot(x, norm.pdf(x, *norm.fit(data['wls_pop'])), color=fp.colors[5])
axes[-1].plot([1 - data['wls_pop'].std(), 1 + data['wls_pop'].std()],
              [axes[-1].get_ylim()[1] * 1.1, axes[-1].get_ylim()[1] * 1.1],
              color=fp.colors[3],
              marker='|')
axes[-1].plot([1 - data['wls_est'][:, 0].std(), 1 + data['wls_est'][:, 0].std()],
              [axes[-1].get_ylim()[1] * 1.1, axes[-1].get_ylim()[1] * 1.1],
              color='#1E5B84',
              marker='|')
axes[-1].set_ylabel(r'$p[\hat{D}^* (D^*)^{-1}]$')
axes[-1].set_yticks([0, 6, 12])
titles.append("b")
axes[-1].set_title("WLS")

axes.append(fig.add_subplot(gs[2, 0], sharex=axes[0]))
axes[-1].hist(data['gls_pop'], bins=bins, density=True, color=fp.colors[5])
axes[-1].plot(x, norm.pdf(x, *norm.fit(data['gls_pop'])), color=fp.colors[5])
axes[-1].plot([1 - data['gls_pop'].std(), 1 + data['gls_pop'].std()],
              [axes[-1].get_ylim()[1] * 1.1, axes[-1].get_ylim()[1] * 1.1],
              color=fp.colors[3],
              marker='|')
axes[-1].plot([1 - data['gls_est'][:, 0].std(), 1 + data['gls_est'][:, 0].std()],
              [axes[-1].get_ylim()[1] * 1.1, axes[-1].get_ylim()[1] * 1.1],
              color='#1E5B84',
              marker='|')
axes[-1].set_xlabel(r'$\hat{D}^* (D^*)^{-1}$')
axes[-1].set_ylabel(r'$p[\hat{D}^* (D^*)^{-1}]$')
axes[-1].set_yticks([0, 12, 24])
titles.append("c")
axes[-1].set_title(" GLS")

axes[-1].plot([], [], color=fp.colors[3], marker='|', label=r'Estimated $\sigma$')
axes[-1].plot([], [], color=fp.NEARLY_BLACK, marker='|', label=r'Population $\sigma$')

fig.align_ylabels(axes)

x_correction = [28] * 3
for i, ax in enumerate(axes):
    x = ax.get_window_extent().x0 - x_correction[i]
    y = ax.get_window_extent().y1 + 11
    x, y = fig.transFigure.inverted().transform([x, y])
    fig.text(x, y, titles[i], ha='left', va='bottom')

plt.figlegend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)

fig.set_size_inches(*figsize)
plt.savefig(paths.figures / "glswlsols.pdf")
plt.close()
