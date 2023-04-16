from enum import Enum


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
