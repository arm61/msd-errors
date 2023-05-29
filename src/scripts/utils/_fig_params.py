"""
Figure parameters for kinisi to help make nice plots.

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

from matplotlib import rcParams

# blue, orange, green, pink, dark green, grey
# colors = ["#0099c8", "#D55E00", "#029E73", "#CC78BC", "#006646", "#949494"]
# colors = ["#5D5779", "#A46355", "#56C1A3", "#FFAABB", "#99DDFF", "#BBBBBB"]    
colors=["#80b8d8", "#80ceb8", "#e9ae7f", "#cca7c6", "#b8b8b8", "#aad9f4"]
FONTSIZE = 7
NEARLY_BLACK = "#161616"
LIGHT_GREY = "#F5F5F5"
WHITE = "#ffffff"
NBINS = 25

MASTER_FORMATTING = {
    "axes.formatter.limits": (-3, 3),
    "xtick.major.pad": 1,
    "ytick.major.pad": 1,
    "ytick.color": NEARLY_BLACK,
    "xtick.color": NEARLY_BLACK,
    "axes.labelcolor": NEARLY_BLACK,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.axisbelow": True,
    "legend.frameon": False,
    'axes.edgecolor': NEARLY_BLACK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": FONTSIZE,
    "font.family": "sans-serif",
    "savefig.bbox": "tight",
    "axes.facecolor": WHITE,
    "axes.labelpad": 2.0,
    "axes.labelsize": FONTSIZE,
    "axes.titlepad": 8,
    "axes.titlesize": FONTSIZE,
    "axes.grid": False,
    "grid.color": WHITE,
    "lines.markersize": 3.0,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "lines.scale_dashes": False,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE,
    "lines.linewidth": 1,
}
for k, v in MASTER_FORMATTING.items():
    rcParams[k] = v


# 510.0pt. \textwidth
# 246.0pt. \columnwidth


def set_size(width: float, fraction: float = 1, subplots: tuple = (1, 1), square: bool = False) -> tuple:
    """Set figure dimensions to avoid scaling in LaTeX.

    :param width: Document width in points, or string of predined document type
    :param fraction: Fraction of the width which you wish the figure to occupy, optional
    :param subplots: The number of rows and columns of subplots, optional
    :param square: Return a square dimension set
    :return: Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    if square:
        return (fig_width_in, fig_width_in)

    return (fig_width_in, [0, 1])
