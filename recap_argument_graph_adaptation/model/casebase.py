from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import recap_argument_graph as ag
from recap_argument_graph_adaptation.controller import convert
from recap_argument_graph_adaptation.model import graph
from recap_argument_graph_adaptation.model.config import Config

log = logging.getLogger(__name__)
config = Config.instance()

metric_keys = {
    "keyword_weight",
    "semantic_similarity",
    "hypernym_proximity",
    "path_similarity",
    "wup_similarity",
}
empty_metrics: t.Callable[[], t.Dict[str, t.Optional[float]]] = lambda: {
    key: None for key in metric_keys
}


@dataclass(frozen=True)
class Concept:
    name: str
    vector: np.ndarray
    pos: t.Optional[POS]
    nodes: t.FrozenSet[graph.AbstractNode]
    metrics: t.Dict[str, t.Optional[float]] = field(default_factory=empty_metrics)

    # @property
    # def best_node(self) -> graph.Node:
    #     return self.nodes[0]

    def __str__(self):
        if self.pos:
            return f"{self.name}/{self.pos.value}"

        return self.name

    def __eq__(self, other: Concept) -> bool:
        return self.name == other.name and self.pos == other.pos

    def __hash__(self) -> int:
        return hash((self.name, self.pos))

    @staticmethod
    def sort(concepts: t.Iterable[Concept]) -> t.List[Concept]:
        return list(sorted(concepts, key=lambda concept: concept.score))

    @property
    def score(self) -> float:
        result = 0
        total_weight = 0

        for metric_name, metric_weight in config.tuning("score").items():
            if (metric := self.metrics[metric_name]) is not None:
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
            "score": self.score,
        }

    @classmethod
    def from_concept(
        cls, source: Concept, metrics: t.Dict[str, t.Optional[float]]
    ) -> Concept:
        return cls(
            source.name,
            source.vector,
            source.pos,
            source.nodes,
            metrics,
        )


def filter_concepts(
    concepts: t.Iterable[Concept],
    min_score: float,
) -> t.Set[Concept]:
    return {concept for concept in concepts if concept.score > min_score}


@dataclass(frozen=True)
class Rule:
    source: Concept
    target: Concept

    def __str__(self) -> str:
        return f"({self.source})->({self.target})"


@dataclass(frozen=True)
class Case:
    relative_path: Path
    query: str
    graph: ag.Graph
    _rules: t.Tuple[Rule, ...]

    def __str__(self) -> str:
        return str(self.relative_path)

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


def spacy2pos(pos: t.Optional[str]) -> t.Optional[POS]:
    if not pos:
        return None

    return {
        "NOUN": POS.NOUN,
        "PROPN": POS.NOUN,
        "VERB": POS.VERB,
        "ADJ": POS.ADJECTIVE,
        "ADV": POS.ADVERB,
    }[pos]


def wn2pos(pos: t.Optional[str]) -> t.Optional[POS]:
    if not pos:
        return None

    return {
        "n": POS.NOUN,
        "v": POS.VERB,
        "a": POS.ADJECTIVE,
        "r": POS.ADVERB,
        "s": POS.ADJECTIVE,
    }.get(pos)


def cn2pos(pos: t.Optional[str]) -> t.Optional[POS]:
    if not pos:
        return None

    return {
        "noun": POS.NOUN,
        "verb": POS.VERB,
        "adjective": POS.ADJECTIVE,
        "adverb": POS.ADVERB,
    }.get(pos)


def pos2wn(pos: t.Optional[POS]) -> t.List[t.Optional[str]]:
    if pos == POS.NOUN:
        return ["n"]
    elif pos == POS.VERB:
        return ["v"]
    elif pos == POS.ADJECTIVE:
        return ["a", "s"]
    elif pos == POS.ADVERB:
        return ["r"]

    return [None]
