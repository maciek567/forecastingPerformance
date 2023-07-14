from enum import Enum


class SeriesColumn(Enum):
    OPEN = "open"
    CLOSE = "close"
    ADJ_CLOSE = "adj_close"
    HIGH = "high"
    LOW = "low"
    VOLUME = "volume"


class DeviationSource(Enum):
    NONE = "no deviations"
    NOISE = "noised"
    INCOMPLETENESS = "incomplete"
    TIMELINESS = "obsolete"


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


def mitigation_short():
    return {
        True: "Y",
        False: "N"
    }