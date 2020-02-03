from enum import Enum


class Method(Enum):
    BETWEEN = "between"
    WITHIN = "within"


class Selector(Enum):
    DIFFERENCE = "difference"
    SIMILARITY = "similarity"
