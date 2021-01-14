from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass
from enum import Enum

import numpy as np
import recap_argument_graph as ag
from recap_argument_graph_adaptation.controller import convert
from recap_argument_graph_adaptation.model import conceptnet as cn
from recap_argument_graph_adaptation.model.config import Config

log = logging.getLogger(__name__)
config = Config.instance()


@dataclass(frozen=True)
class Concept:
    name: str
    vector: np.ndarray
    pos: POS  # needed as it might be the case that the rule specifies a pos that is not available in ConceptNet.
    nodes: t.Tuple[cn.Node, ...]
    synsets: t.Tuple[str, ...]
    keyword_weight: t.Optional[float]
    semantic_similarity: t.Optional[float] = None
    conceptnet_path_distance: t.Optional[float] = None
    wordnet_path_similarity: t.Optional[float] = None
    wordnet_wup_similarity: t.Optional[float] = None

    @property
    def best_node(self) -> cn.Node:
        return self.nodes[0]

    def __str__(self):
        if self.pos != POS.OTHER:
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
        rules_limit = config.tuning("rules")["adaptation_limit"]
        slice = len(self._rules) if rules_limit == 0 else rules_limit

        return self._rules[:slice]

    @property
    def benchmark_rules(self) -> t.Tuple[Rule, ...]:
        return self._rules


@dataclass(frozen=True)
class Evaluation:
    score: float
    benchmark_and_computed: t.Set[Concept]
    only_benchmark: t.Set[Concept]
    only_computed: t.Set[Concept]

    def to_dict(self, compact: bool = False) -> t.Dict[str, t.Any]:
        if compact:
            return {
                "score": self.score,
                "benchmark_and_computed": len(self.benchmark_and_computed),
                "only_benchmark": len(self.only_benchmark),
                "only_computed": len(self.only_computed),
            }

        return {
            "score": self.score,
            "benchmark_and_computed": convert.list_str(self.benchmark_and_computed),
            "only_benchmark": convert.list_str(self.only_benchmark),
            "only_computed": convert.list_str(self.only_computed),
        }

    def __lt__(self, other):
        return self.score < other.score

    def __le__(self, other):
        return self.score <= other.score

    def __gt__(self, other):
        return self.score > other.score

    def __ge__(self, other):
        return self.score >= other.score


class POS(Enum):
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    OTHER = "other"


spacy_pos_mapping = {
    "NOUN": POS.NOUN,
    "PROPN": POS.NOUN,
    "VERB": POS.VERB,
    "ADJ": POS.ADJECTIVE,
    "ADV": POS.ADVERB,
}

wn_pos_mapping = {
    "n": POS.NOUN,
    "v": POS.VERB,
    "a": POS.ADJECTIVE,
    "r": POS.ADVERB,
    "s": POS.ADJECTIVE,
}


def wn_pos(pos: POS) -> t.List[str]:
    if pos == POS.NOUN:
        return ["n"]
    elif pos == POS.VERB:
        return ["v"]
    elif pos == POS.ADJECTIVE:
        return ["a", "s"]
    elif pos == POS.ADVERB:
        return ["r"]

    return []
