import os
import time
from enum import Enum

import matplotlib
import matplotlib.font_manager
from pandas import Series


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


class Mitigation(Enum):
    DATA = "data"
    TIME = "time"


def perform_mitigation(series: Series, mitigation_method, multiple_runs: bool = False) -> dict:
    start_time = time.time_ns()
    mitigated_series = mitigation_method(series)
    elapsed_time_ms = 0.0

    if not multiple_runs:
        elapsed_time_ms = (time.time_ns() - start_time) / 1e6
    if multiple_runs:
        for i in range(99):
            mitigation_method(series)
        elapsed_time_ms = (time.time_ns() - start_time) / 1e6 / 100

    return {Mitigation.DATA: mitigated_series, Mitigation.TIME: elapsed_time_ms}


def set_legend(ax):
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.2))
    legend.get_frame().set_alpha(0.5)


def save_image(plt, title):
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.serif"] = "courier"
    matplotlib.rcParams["font.size"] = 11

    path = os.path.join('..', 'data', 'graphs', title.replace(' ', '_').replace(',', ''))
    plt.savefig(f"{path}.pdf", bbox_inches='tight')


def set_ticks_size(ax, axis, size):
    ax[0].tick_params(axis=axis, labelsize=size)
    ax[1].tick_params(axis=axis, labelsize=size)
    ax[2].tick_params(axis=axis, labelsize=size)
