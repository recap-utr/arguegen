from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass

import numpy as np
import recap_argument_graph as ag
from recap_argument_graph_adaptation.model import graph
from recap_argument_graph_adaptation.model.config import config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Concept:
    name: str
    vector: np.ndarray
    pos: graph.POS  # needed as it might be the case that the rule specifies a pos that is not available in ConceptNet.
    nodes: t.Tuple[graph.Node, ...]
    synsets: t.Tuple[str, ...]
    keyword_weight: t.Optional[float]
    semantic_similarity: t.Optional[float] = None
    conceptnet_path_distance: t.Optional[float] = None
    wordnet_path_similarity: t.Optional[float] = None
    wordnet_wup_similarity: t.Optional[float] = None

    @property
    def best_node(self) -> graph.Node:
        return self.nodes[0]

    def __str__(self):
        if self.pos != graph.POS.OTHER:
            return f"{self.name}/{self.pos.value}"

        return self.name

    def __eq__(self, other: Concept) -> bool:
        return self.name == other.name and self.pos == other.pos

    def __hash__(self) -> int:
        return hash((self.name, self.pos))

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
        }

        result = 0
        total_weight = 0

        for metric_name, metric_weight in config.tuning("score").items():
            if (metric := metrics[metric_name]) is not None:
                result += metric * metric_weight
                total_weight += metric_weight

        # If one metric is not set, the weights would not sum to 1.
        # Thus, the result is normalized.
        if total_weight > 0:
            return result * (1 / total_weight)

        return 0.0

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            "concept": str(self),
            "nodes": [str(node) for node in self.nodes],
            "synsets": self.synsets,
            "score": self.score,
        }

    @classmethod
    def from_concept(
        cls,
        source: Concept,
        metrics: t.Tuple[t.Optional[float], ...],
    ) -> Concept:
        return Concept(
            source.name,
            source.vector,
            source.pos,
            source.nodes,
            source.synsets,
            *metrics,
        )


def _dist2sim(distance: t.Optional[float]) -> t.Optional[float]:
    if distance is not None:
        return 1 / (1 + distance)

    return None


@dataclass(frozen=True)
class Rule:
    source: Concept
    target: Concept

    def __str__(self) -> str:
        return f"({self.source})->({self.target})"


rules_limit = config.tuning("rules")["adaptation_limit"]


@dataclass(frozen=True)
class Case:
    name: str
    query: str
    graph: ag.Graph
    _rules: t.Tuple[Rule, ...]

    def __str__(self) -> str:
        return self.name

    @property
    def rules(self) -> t.Tuple[Rule, ...]:
        current_limit = len(self._rules) if rules_limit == 0 else rules_limit

        return self._rules[:current_limit]

    @property
    def benchmark_rules(self) -> t.Tuple[Rule, ...]:
        return self._rules
