from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field
from enum import Enum

import recap_argument_graph as ag
from recap_argument_graph_adaptation.model import graph
from recap_argument_graph_adaptation.model.config import config
from spacy.tokens import Doc  # type: ignore

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

    def __eq__(self, other: Concept) -> bool:
        return self.name.text == other.name.text and self.pos == other.pos

    def __hash__(self) -> int:
        return hash((self.name.text, self.pos))

    @staticmethod
    def only_relevant(concepts: t.Iterable[Concept]) -> t.Set[Concept]:
        return {
            concept
            for concept in concepts
            if (
                concept.conceptual_distance
                < config["conceptnet"]["node"]["max_conceptual_distance_and"]
                and concept.semantic_similarity
                > config["conceptnet"]["node"]["min_semantic_similarity_and"]
            )
            or concept.conceptual_distance
            < config["conceptnet"]["node"]["max_conceptual_distance_or"]
            or concept.semantic_similarity
            > config["conceptnet"]["node"]["min_semantic_similarity_or"]
        }


@dataclass
class Rule:
    source: Concept
    target: Concept

    def __str__(self) -> str:
        return f"({self.source})->({self.target})"


@dataclass(frozen=True)
class Case:
    name: str
    query: str
    graph: ag.Graph
    rules: t.List[Rule]
    benchmark_graph: ag.Graph
    benchmark_rules: t.List[Rule]
