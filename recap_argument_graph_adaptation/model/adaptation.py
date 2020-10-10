import logging
import typing as t
from dataclasses import dataclass, field
from enum import Enum

import recap_argument_graph as ag
from spacy.tokens import Doc

from recap_argument_graph_adaptation.model import graph
from recap_argument_graph_adaptation.model.config import config

log = logging.getLogger(__name__)


class Method(Enum):
    BETWEEN = "between"
    WITHIN = "within"


class Selector(Enum):
    DIFFERENCE = "difference"
    SIMILARITY = "similarity"


@dataclass(frozen=True)
class Concept:
    name: Doc
    pos: graph.POS  # needed as it might be the case that the rule specifies a pos that is not available in ConceptNet.
    nodes: t.Tuple[graph.Node, ...]
    semantic_similarity: float
    conceptual_distance: int

    @property
    def best_node(self) -> graph.Node:
        return self.nodes[0]

    def __str__(self):
        if self.pos != graph.POS.OTHER:
            return f"{self.name}/{self.pos.value}"

        return self.name


@dataclass
class Rule:
    source: Concept
    target: Concept

    def __str__(self) -> str:
        return f"({self.source})->({self.target}) [nodes ({self.source.node})->({self.target.node})]"


@dataclass(frozen=True)
class Case:
    name: str
    query: str
    graph: ag.Graph
    rules: t.List[Rule]
    benchmark_graph: ag.Graph
    benchmark_rules: t.List[Rule]
