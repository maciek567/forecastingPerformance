import os
from enum import Enum

import matplotlib
import matplotlib.font_manager


class SeriesColumn(Enum):
    OPEN = "open"
    CLOSE = "close"
    ADJ_CLOSE = "adj_close"
    HIGH = "high"
    LOW = "low"
    VOLUME = "volume"


class DeviationSource(Enum):
    NONE = "no deviations"
    NOISE = "noise"
    INCOMPLETENESS = "incompleteness"
    TIMELINESS = "obsolescence"


def sources_short():
    return {
        DeviationSource.NONE: "-",
        DeviationSource.NOISE: "N",
        DeviationSource.INCOMPLETENESS: "I",
        DeviationSource.TIMELINESS: "T"
    }


class DeviationScale(Enum):
    SLIGHTLY = "slightly"
    MODERATELY = "moderately"
    HIGHLY = "highly"


def scales_short():
    return {
        None: "-",
        DeviationScale.SLIGHTLY: "S",
        DeviationScale.MODERATELY: "M",
        DeviationScale.HIGHLY: "H",
    }


def mitigation_short():
    return {
        True: "Y",
        False: "N"
    }


class DeviationRange(Enum):
    ALL = "all"
    PARTIAL = "partially"


class Deviation:
    def __init__(self, method, scale):
        self.method = method
        self.scale = scale


def set_legend(ax):
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.2))
    legend.get_frame().set_alpha(0.5)


def save_image(plt, title):
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = "courier"
    matplotlib.rcParams["font.size"] = 11

    path = os.path.join('..', 'data', 'graphs', title.replace(' ', '_').replace(',', ''))
    plt.savefig(f"{path}.pdf", bbox_inches='tight')


def set_ticks_size(ax, axis, size):
    ax[0].tick_params(axis=axis, labelsize=size)
    ax[1].tick_params(axis=axis, labelsize=size)
    ax[2].tick_params(axis=axis, labelsize=size)
