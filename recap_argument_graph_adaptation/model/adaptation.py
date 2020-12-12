from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field
from enum import Enum
from nltk.corpus.reader.wordnet import Synset

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
    synsets: t.Tuple[Synset, ...]
    semantic_similarity: t.Optional[float] = None
    conceptnet_path_distance: t.Optional[float] = None
    wordnet_path_similarity: t.Optional[float] = None
    wordnet_wup_similarity: t.Optional[float] = None
    wordnet_path_distance: t.Optional[float] = None
    keyword_weight: t.Optional[float] = None

    @property
    def best_node(self) -> graph.Node:
        return self.nodes[0]

    def __str__(self):
        if self.pos != graph.POS.OTHER:
            return f"{self.name.text}/{self.pos.value}"

        return self.name.text

    def __eq__(self, other: Concept) -> bool:
        return self.name.text == other.name.text and self.pos == other.pos

    def __hash__(self) -> int:
        return hash((self.name.text, self.pos))

    @staticmethod
    def only_relevant(
        concepts: t.Iterable[Concept],
        min_score: float,
    ) -> t.Set[Concept]:
        return {concept for concept in concepts if concept.score > min_score}

    @staticmethod
    def sort(concepts: t.Iterable[Concept]) -> t.List[Concept]:
        return list(sorted(concepts, key=lambda concept: concept.score))

    @property
    def score(self) -> float:
        result = 0
        total_metrics = 0

        for sim in (self.keyword_weight, self.semantic_similarity):
            if sim is not None:
                result += sim
                total_metrics += 1

        for dist in (self.wordnet_path_distance, self.conceptnet_path_distance):
            if dist is not None:
                result += _sim(dist)
                total_metrics += 1

        return result / total_metrics

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            "concept": str(self),
            "nodes": [str(node) for node in self.nodes],
            "synsets": [synset.name() for synset in self.synsets],
            "score": self.score,
        }


def _sim(distance: int) -> float:
    return 1 / (1 + distance)


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
