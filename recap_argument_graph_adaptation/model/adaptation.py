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


@dataclass(frozen=True)
class Case:
    name: str
    query: str
    graph: ag.Graph
    rules: t.List[Rule]
    benchmark_graph: ag.Graph
    benchmark_rules: t.List[Rule]


@dataclass(frozen=True)
class Concept:
    original_name: str
    conceptnet_name: str

    def __str__(self):
        return self.conceptnet_name
