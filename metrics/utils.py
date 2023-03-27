from enum import Enum


class MetricLevel(Enum):
    VALUES = "values"
    TUPLES = "tuples"
    RELATION = "relation"


class Strength(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class Incomplete(Enum):
    SLIGHTLY = "slightly"
    MODERATELY = "moderately"
    HIGHLY = "highly"


class DefectionScale(Enum):
    NONE = "undamaged"
    SLIGHTLY = "slightly"
    MODERATELY = "moderately"
    HIGHLY = "highly"


class DefectionRange(Enum):
    ALL = "all"
    PARTIAL = "partially"


class DefectsSource(Enum):
    NONE = "no defects"
    NOISE = "noise"
    INCOMPLETENESS = "incompleteness"
