import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from uncertainties import ufloat
from tueplots import figsizes

import utils._fig_params as fp
import paths

lengths = [16, 32, 64, 128, 256, 512, 1024]

figsize = figsizes.icml2022_half(nrows=1, ncols=1, height_to_width_ratio=0.5)['figure.figsize']
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(1, 1, figure=fig)

axes = []
titles = []
cap = ['True']
mpl.rcParams["axes.spines.right"] = True

i = 0
data = np.zeros((len(lengths), 2, 3200))
for ll, length in enumerate(lengths):
    d = np.load(paths.data / f"random_walks/numerical/D_1_128_{length}.npz")
    data[ll, 0] = d["diff_c"]
    data[ll, 1] = d["intercept"]

axes.append(fig.add_subplot(gs[0, 0]))
axes[-1].errorbar(
    lengths,
    data[:, 0].mean(-1),
    data[:, 0].std(-1) * 2,
    marker=".",
    #   ls="-",
    lw=0.5,
    color=fp.colors[0])
axes[-1].set_xscale("log", base=2)
axes[-1].set_xlabel("Simulation Length / s")
axes[-1].set_ylabel("$\hat{D}^*$ / m$^{2}$s$^{-1}$")
axes[-1].yaxis.label.set_color(fp.colors[0])
axes[-1].tick_params(axis='y', colors=fp.colors[0])
axes[-1].set_xticks([32, 128, 512])
axes[-1].set_xticklabels([32, 128, 512])
axes[-1].minorticks_off()

axes.append(axes[-1].twinx())
axes[-1].errorbar(
    lengths,
    data[:, 1].mean(-1),
    data[:, 1].std(-1) * 2,
    marker=".",
    #   ls="--",
    lw=0.5,
    color=fp.colors[1])
axes[-1].set_xscale("log", base=2)
axes[-1].set_xlabel("Simulation Length / s")
axes[-1].set_ylabel(r"$\hat{D}_{\mathrm{offset}}$ / m$^{2}$")
axes[-1].yaxis.label.set_color(fp.colors[1])
axes[-1].tick_params(axis='y', colors=fp.colors[1])
axes[-1].set_xticks([32, 128, 512])
axes[-1].set_xticklabels([32, 128, 512])
axes[-1].minorticks_off()

v = ufloat(*norm.fit(data[-1, 0]))
write_out = open(paths.output / f"D_true_1.txt", "w")
write_out.write(r"\SI{" + f"{v:L}" + r"}{\meter^2\per\second}")
write_out.close()
v = ufloat(*norm.fit(data[-1, 1]))
write_out = open(paths.output / f"D0_true_1.txt", "w")
write_out.write(r"\SI{" + f"{v:.2fL}" + r"}{\meter^2}")
write_out.close()

legend = plt.figlegend(['$\hat{D}^*$', r'$\hat{D}_{\mathrm{offset}}$'],
                       loc='upper center',
                       bbox_to_anchor=(0.5, -0.05),
                       ncol=2)

plt.savefig(paths.figures / "longtime.pdf", bbox_inches="tight")
plt.close()
