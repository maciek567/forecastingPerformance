from enum import Enum
import os
import matplotlib.font_manager
import matplotlib


class SeriesColumn(Enum):
    OPEN = "open"
    CLOSE = "close"
    ADJ_CLOSE = "adjclose"
    HIGH = "high"
    LOW = "low"
    VOLUME = "volume"


class DeviationSource(Enum):
    NONE = "no deviations"
    NOISE = "noise"
    INCOMPLETENESS = "incompleteness"
    TIMELINESS = "obsolescence"


class DeviationScale(Enum):
    SLIGHTLY = "slightly"
    MODERATELY = "moderately"
    HIGHLY = "highly"


class DeviationRange(Enum):
    ALL = "all"
    PARTIAL = "partially"


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
