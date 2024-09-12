import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from uravu.distribution import Distribution
from uravu.axis import Axis

import utils._fig_params as fp
from utils.plotting_helper import mid_points
import paths

correlation = 'true'
jump = 1


def find_exp(number) -> int:
    base10 = np.log10(abs(number))
    return int(abs(np.floor(base10)))


def f(x, a, b):
    return b * x**a


xaxis = np.array([16, 32, 64, 128, 256, 512, 1024])

figsize = (3.66, 1.5)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.6, hspace=0.4)

x = np.linspace(xaxis.min(), xaxis.max(), 1000)

axes = []
titles = []
letters = ['a', 'b']

data = np.load(paths.data / f"random_walks/stat_eff.npz")

axes.append(fig.add_subplot(gs[0, 0]))
axes[-1].plot(xaxis, data['ky'], marker='.', ls="", color='#67B99B')
k = f(x, data['ka'], data['kb'])
axes[-1].plot(x, k, color='#67B99B')
axes[-1].plot(xaxis, data['wy'], marker='.', ls="", color='#81A8D9')
k = f(x, data['wa'], data['wb'])
axes[-1].plot(x, k, color='#81A8D9')
axes[-1].plot(xaxis, data['py'], marker='.', ls="", color='#F3ADBC')
k = f(x, data['pa'], data['pb'])
axes[-1].plot(x, k, color='#F3ADBC')
axes[-1].plot(xaxis, data['ny'], marker='.', ls="", color='#E08D6D')
k = f(x, data['na'], data['nb'])
axes[-1].plot(x, k, color='#E08D6D')
axes[-1].set_xscale('log', base=2)
axes[-1].set_yscale('log', base=2)
axes[-1].set_xlabel(r'$N_{{\mathrm{{atoms}}}}$')
axes[-1].set_ylabel(r"$\sigma^2 [\hat{D}^*]$")
titles.append(f"{letters[0]}")
axes[-1].set_title(r"fixed $t_{\max} = 128$")
axes[-1].set_xticks([32, 128, 512])
axes[-1].set_xticklabels([32, 128, 512])
axes[-1].set_xlim(8, 2048)
axes[-1].set_yticks([2**-15, 2**-12, 2**-9, 2**-6, 2**-3])
axes[-1].set_ylim(2**-17, 2**-3)
axes[-1].minorticks_off()

axes.append(fig.add_subplot(gs[0, 1], sharey=axes[-1]))
axes[-1].plot(xaxis, data['ky2'], marker='.', ls="", color='#67B99B')
k = f(x, data['ka2'], data['kb2'])
axes[-1].plot(x, k, color='#67B99B')
axes[-1].plot(xaxis, data['wy2'], marker='.', ls="", color='#81A8D9')
k = f(x, data['wa2'], data['wb2'])
axes[-1].plot(x, k, color='#81A8D9')
axes[-1].plot(xaxis, data['py2'], marker='.', ls="", color='#F3ADBC')
k = f(x, data['pa2'], data['pb2'])
axes[-1].plot(x, k, color='#F3ADBC')
axes[-1].plot(xaxis, data['ny2'], marker='.', ls="", color='#E08D6D')
k = f(x, data['na2'], data['nb2'])
axes[-1].plot(x, k, color='#E08D6D')
axes[-1].set_xscale('log', base=2)
axes[-1].set_xlabel(r'$\Delta t_{\mathrm{max}}$')
axes[-1].set_ylabel(r"$\sigma^2 [\hat{D}^*]$")
titles.append(f"{letters[1]}")
axes[-1].set_title(r"fixed $N_{\mathrm{atoms}} = 128$")
axes[-1].set_xticks([32, 128, 512])
axes[-1].set_xticklabels([32, 128, 512])
axes[-1].set_xlim(8, 2048)
axes[-1].set_yticks([2**-15, 2**-12, 2**-9, 2**-6, 2**-3])
axes[-1].minorticks_off()

axes[-1].plot([], [], marker='.', ls='-', color='#F3ADBC', label='OLS')
axes[-1].plot([], [], marker='.', ls='-', color='#81A8D9', label='WLS($\sigma_{\mathrm{num}}$)')
axes[-1].plot([], [], marker='.', ls='-', color='#67B99B', label='approximate Bayesian regression ($\Sigma^\prime$)')
axes[-1].plot([], [], marker='.', ls='-', color='#E08D6D', label='GLS($\Sigma_{\mathrm{num}}$)')

plt.figlegend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1)

x_correction = [36] * 2
for i, ax in enumerate(axes):
    x = ax.get_window_extent().x0 - x_correction[i]
    y = ax.get_window_extent().y1 + 11.5
    x, y = fig.transFigure.inverted().transform([x, y])
    f = fig.text(x, y, titles[i], ha='left')

for i, ax in enumerate(axes):
    print(ax.get_window_extent().x1 - ax.get_window_extent().x0)
    print(ax.get_window_extent().y1 - ax.get_window_extent().y0)

plt.savefig(paths.figures / "stat_eff.pdf", bbox_inches="tight")
plt.close()
