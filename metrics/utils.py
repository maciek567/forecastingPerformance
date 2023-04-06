from enum import Enum


class MetricLevel(Enum):
    VALUES = "values"
    TUPLES = "tuples"
    RELATION = "relation"


class DefectsSource(Enum):
    NONE = "no defects"
    NOISE = "noise"
    INCOMPLETENESS = "incompleteness"


class DefectsScale(Enum):
    SLIGHTLY = "slightly"
    MODERATELY = "moderately"
    HIGHLY = "highly"


class DefectionRange(Enum):
    ALL = "all"
    PARTIAL = "partially"
