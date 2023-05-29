from uncertainties import ufloat
from uravu.distribution import Distribution

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
CREDIBLE_INTERVALS = [[15.86555, 84.13445, 1], [2.27500, 97.725, 0.6], [0.135, 99.865, 0.2]]


def to_string(d: Distribution, units: str = None, exponent: int = 0, normal_override: bool = False) -> str:
    """
    Turns Distribution objects to LaTeX strings

    :param d: Distribution to LaTeXify.
    :param units: Units to include.
    """
    if normal_override or d.normal:
        starter = 'num'
        if units:
            starter = 'SI'

        if exponent != 0:
            e = Distribution(d.samples / (10**exponent))
            v = ufloat(e.samples.mean(), e.samples.std(ddof=1) * 1.96)
            s = "$\\" + f"{starter}" + r"{" + f"{v:.2fL}E{exponent}" + r"}"
        else:
            v = ufloat(d.samples.mean(), d.samples.std(ddof=1) * 1.96)
            s = "$\\" + f"{starter}" + r"{" + f"{v:.2fL}" + r"}"
        if units is not None:
            s += r"{" + f"{units}" + r"}$"
        else:
            s += '$'
    else:
        if exponent != 0:
            s = r"$\big" + f"[({d.dist_max / (10 ** exponent):.2f}^" \
                + r"{+" + f"{(d.con_int()[1] - d.dist_max) / (10 ** exponent):.2f}" \
                + r"}_{-" + f"{(d.dist_max - d.con_int()[0]) / (10 ** exponent):.2f}" \
                + r"})\times 10^{" + f"{exponent}" + r"}\big]"
        else:
            s = r"$" + f"({d.dist_max:.2f}^" + r"{+" \
                + f"{(d.con_int()[1] - d.dist_max):.2f}" + r"}_{-" \
                + f"{(d.dist_max - d.con_int()[0]):.2f}" + r"})"
        if units is not None:
            s += r"\,\si{" + f"{units}" + r"}$"
        else:
            s += '$'
    return s


def mid_points(axis):
    """
    Find the mid point in an axis, in axes space.
    """
    x = (axis.get_window_extent().x1 - axis.get_window_extent().x0) / 2 + axis.get_window_extent().x0
    y = (axis.get_window_extent().y1 - axis.get_window_extent().y0) / 2 + axis.get_window_extent().y0
    return x, y

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))