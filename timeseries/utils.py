from enum import Enum


class SeriesColumn(Enum):
    OPEN = "open"
    CLOSE = "close"
    ADJ_CLOSE = "adjclose"
    HIGH = "high"
    LOW = "low"
    VOLUME = "volume"


class DefectsSource(Enum):
    NONE = "no defects"
    NOISE = "noise"
    INCOMPLETENESS = "incompleteness"
    TIMELINESS = "obsolescence"


class DefectsScale(Enum):
    SLIGHTLY = "slightly"
    MODERATELY = "moderately"
    HIGHLY = "highly"


class DefectionRange(Enum):
    ALL = "all"
    PARTIAL = "partially"
