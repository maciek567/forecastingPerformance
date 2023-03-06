from enum import Enum


class MetricLevel(Enum):
    VALUES = "values"
    TUPLES = "tuples"
    RELATION = "relation"


class QualityDifferencesSource(Enum):
    NOISE_STRENGTH = "noise strength"
    SENSITIVENESS = "sensitiveness"
    INCOMPLETENESS = "incompleteness"


class Strength(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class Sensitiveness(Enum):
    SENSITIVE = "sensitive"
    MODERATE = "moderate"
    INSENSITIVE = "insensitive"


class Incomplete(Enum):
    SLIGHTLY = 1
    MODERATELY = 2
    HIGHLY = 3
