from __future__ import annotations

import logging
from multiprocessing import Value
import statistics
import typing as t
from dataclasses import dataclass, field
from enum import Enum

import recap_argument_graph as ag
from nltk.corpus.reader.wordnet import Synset
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
    keyword_weight: t.Optional[float]
    semantic_similarity: t.Optional[float] = None
    conceptnet_path_distance: t.Optional[float] = None
    wordnet_path_similarity: t.Optional[float] = None
    wordnet_wup_similarity: t.Optional[float] = None
    wordnet_path_distance: t.Optional[float] = None

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
        metrics = {
            "keyword_weight": self.keyword_weight,
            "semantic_similarity": self.semantic_similarity,
            "conceptnet_path_distance": _dist2sim(self.conceptnet_path_distance),
            "wordnet_path_similarity": self.wordnet_path_similarity,
            "wordnet_wup_similarity": self.wordnet_wup_similarity,
            "wordnet_path_distance": _dist2sim(self.wordnet_path_distance),
        }

        if round(sum(config["nlp"]["concept_score"].values()), 2) != 1:
            raise ValueError("The sum is not 1.")

        result = 0
        total_weight = 0

        for metric_name, metric_weight in config["nlp"]["concept_score"].items():
            if (metric := metrics[metric_name]) is not None:
                result += metric * metric_weight
                total_weight += metric_weight

        # If one metric is not set, the weights would not sum to 1.
        # Thus, the result is normalized.
        return result * (1 / total_weight)

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            "concept": str(self),
            "nodes": [str(node) for node in self.nodes],
            "synsets": [synset.name() for synset in self.synsets],
            "score": self.score,
        }


def _dist2sim(distance: t.Optional[float]) -> t.Optional[float]:
    if distance is not None:
        return 1 / (1 + distance)

    return None


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
