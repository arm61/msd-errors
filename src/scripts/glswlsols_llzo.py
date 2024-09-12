import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
from scipy.stats import norm
from uravu.distribution import Distribution

import utils._fig_params as fp
import paths

length = 10000

data = np.load(f'src/data/llzo/glswlsols_10_{length}.npz')

figsize = (3.744, 3.4)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.8)

x = np.linspace(data['gls_pop'].mean() - data['ols_pop'].std() * 3, data['gls_pop'].mean() + data['ols_pop'].std() * 3, 1000)

axes = []
titles = []
axes.append(fig.add_subplot(gs[0, 0]))
bins = np.linspace((data['ols_pop']).min(), (data['ols_pop']).max(), fp.NBINS)
axes[-1].hist(data['ols_pop'], bins=bins, density=True, color='#ECBCA7')
axes[-1].plot(x, norm.pdf(x, *norm.fit(data['ols_pop'])), color='#B8B8B8')
axes[-1].plot([data['gls_pop'].mean() - data['ols_est'], data['gls_pop'].mean() + data['ols_est']],
              [axes[-1].get_ylim()[1] * 1.1, axes[-1].get_ylim()[1] * 1.1],
              color='#305980',
              marker='|')
axes[-1].plot([data['gls_pop'].mean() - data['ols_pop'].std(), data['gls_pop'].mean() + data['ols_pop'].std()],
              [axes[-1].get_ylim()[1] * 1.1, axes[-1].get_ylim()[1] * 1.1],
              color='#F3ADBC',
              marker='|')
# axes[-1].set_xticks([0.8, 0.9, 1, 1.1, 1.2])
# axes[-1].set_xlim(0.75, 1.25)
# axes[-1].set_yticks([0, 3, 6])
axes[-1].set_ylabel(r'$p(\hat{D}^*)$')
titles.append("a")
axes[-1].set_title("ordinary least squares")

axes.append(fig.add_subplot(gs[1, 0], sharex=axes[0]))
axes[-1].hist(data['wls_pop'], bins=bins, density=True, color='#ECBCA7')
axes[-1].plot(x, norm.pdf(x, *norm.fit(data['wls_pop'])), color='#B8B8B8')
axes[-1].plot([data['gls_pop'].mean() - data['wls_est'], data['gls_pop'].mean() + data['wls_est']],
              [axes[-1].get_ylim()[1] * 1.1, axes[-1].get_ylim()[1] * 1.1],
              color='#305980',
              marker='|')
axes[-1].plot([data['gls_pop'].mean() - data['wls_pop'].std(), data['gls_pop'].mean() + data['wls_pop'].std()],
              [axes[-1].get_ylim()[1] * 1.1, axes[-1].get_ylim()[1] * 1.1],
              color='#F3ADBC',
              marker='|')
axes[-1].set_ylabel(r'$p(\hat{D}^*)$')
# axes[-1].set_yticks([0, 6, 12])
titles.append("b")
axes[-1].set_title("weighted least squares")

axes.append(fig.add_subplot(gs[2, 0], sharex=axes[0]))
axes[-1].hist(data['gls_pop'], bins=bins, density=True, color='#ECBCA7')
axes[-1].plot(x, norm.pdf(x, *norm.fit(data['gls_pop'])), color='#B8B8B8')
axes[-1].plot([data['gls_pop'].mean() - data['gls_est'], data['gls_pop'].mean() + data['gls_est']],
              [axes[-1].get_ylim()[1] * 1.1, axes[-1].get_ylim()[1] * 1.1],
              color='#305980',
              marker='|')
axes[-1].plot([data['gls_pop'].mean() - data['gls_pop'].std(), data['gls_pop'].mean() + data['gls_pop'].std()],
              [axes[-1].get_ylim()[1] * 1.1, axes[-1].get_ylim()[1] * 1.1],
              color='#F3ADBC',
              marker='|')
axes[-1].set_xlabel(r'$\hat{D}^*$')
axes[-1].set_ylabel(r'$p(\hat{D}^*)$')
# axes[-1].set_yticks([0, 12, 24])
titles.append("c")
axes[-1].set_title("generalised least squares")

axes[-1].plot([], [], color='#F3ADBC', marker='|', label=r'population $\sigma$')
axes[-1].plot([], [], color='#305980', marker='|', label=r'mean estimated $\sigma$')

fig.align_ylabels(axes)

x_correction = [28] * 3
for i, ax in enumerate(axes):
    x = ax.get_window_extent().x0 - x_correction[i]
    y = ax.get_window_extent().y1 + 11
    x, y = fig.transFigure.inverted().transform([x, y])
    fig.text(x, y, titles[i], ha='left', va='bottom')

plt.figlegend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)

fig.set_size_inches(*figsize)
plt.savefig(paths.figures / "glswlsols_llzo.pdf")
plt.close()
