import recap_argument_graph as ag
from dataclasses import dataclass
from enum import Enum
import typing as t


class Method(Enum):
    BETWEEN = "between"
    WITHIN = "within"


class Selector(Enum):
    DIFFERENCE = "difference"
    SIMILARITY = "similarity"


Rule = t.Tuple[str, str]


@dataclass
class Case:
    graph: ag.Graph
    rules: t.List[Rule]
