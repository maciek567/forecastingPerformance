from enum import Enum


class SeriesColumn(Enum):
    OPEN = "open"
    CLOSE = "close"
    ADJ_CLOSE = "adjclose"
    HIGH = "high"
    LOW = "low"
    VOLUME = "volume"
