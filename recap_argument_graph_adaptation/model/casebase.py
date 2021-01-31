from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import recap_argument_graph as ag
from recap_argument_graph_adaptation.controller import convert
from recap_argument_graph_adaptation.model import graph, spacy
from recap_argument_graph_adaptation.model.config import Config

log = logging.getLogger(__name__)
config = Config.instance()

metric_keys = {
    "keyword_weight",
    "nodes_semantic_similarity",
    "concept_semantic_similarity",
    "hypernym_proximity",
    "major_claim_proximity",
    "nodes_path_similarity",
    "nodes_wup_similarity",
    "query_nodes_semantic_similarity",
    "query_concept_semantic_similarity",
}
empty_metrics: t.Callable[[], t.Dict[str, t.Optional[float]]] = lambda: {
    key: None for key in metric_keys
}


class ArgumentNode(ag.Node):
    __slots__ = ("vector",)

    vector: spacy.Vector

    def __post_init__(self):
        self.vector = spacy.vector(self.plain_text)

    def __str__(self) -> str:
        return str(self.key)

    def __eq__(self, other: ArgumentNode) -> bool:
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)


@dataclass(frozen=True)
class Concept:
    name: str
    vector: spacy.Vector
    forms: t.FrozenSet[str]
    pos: t.Optional[POS]
    inodes: t.FrozenSet[ArgumentNode]
    nodes: t.FrozenSet[graph.AbstractNode]
    metrics: t.Dict[str, t.Optional[float]] = field(default_factory=empty_metrics)

    def __str__(self):
        code = self.code

        if self.inodes:
            code += f"/{set(inode.key for inode in self.inodes)}"

        return code

    def __eq__(self, other: Concept) -> bool:
        return (
            self.name == other.name
            and self.pos == other.pos
            and self.forms == other.forms
            and self.inodes == other.inodes
        )

    # TODO: If the pos is missing (None), the forms will differ. We need to account for that!
    def __hash__(self) -> int:
        return hash((self.name, self.pos, self.forms, self.inodes))

    def part_eq(self, other: Concept) -> bool:
        return self.pos == other.pos and self.inodes == other.inodes

    @property
    def code(self) -> str:
        out = f"{self.name}"

        if self.pos:
            out += f"/{self.pos.value}"

        return out

    @property
    def inode_vectors(self) -> t.List[spacy.Vector]:
        return [inode.vector for inode in self.inodes]

    @staticmethod
    def sort(concepts: t.Iterable[Concept]) -> t.List[Concept]:
        return list(sorted(concepts, key=lambda concept: concept.score))

    @property
    def score(self) -> float:
        return score(self.metrics)

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            "concept": str(self),
            "forms": str(self.forms),
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
            source.forms,
            source.pos,
            source.inodes,
            source.nodes,
            metrics,
        )


def score(metrics: t.Dict[str, t.Optional[float]]) -> float:
    result = 0
    total_weight = 0

    for metric_name, metric_weight in config.tuning("score").items():
        if (metric := metrics[metric_name]) is not None:
            result += metric * metric_weight
            total_weight += metric_weight

    # Normalize the result.
    if total_weight > 0:
        return result / total_weight

    return 0.0


def filter_concepts(
    concepts: t.Iterable[Concept],
    min_score: float,
) -> t.Set[Concept]:
    return {concept for concept in concepts if concept.score > min_score}


@dataclass(frozen=True, eq=True)
class Rule:
    source: Concept
    target: Concept

    def __str__(self) -> str:
        return f"({self.source})->({self.target})"


@dataclass(frozen=True)
class UserQuery:
    text: str
    vector: spacy.Vector

    def __str__(self) -> str:
        return self.text

    def __eq__(self, other: UserQuery) -> bool:
        return self.text == other.text

    def __hash__(self) -> int:
        return hash(self.text)


@dataclass(frozen=True)
class Case:
    relative_path: Path
    user_query: UserQuery
    graph: ag.Graph
    _rules: t.Tuple[Rule, ...]

    def __str__(self) -> str:
        return str(self.relative_path)

    @property
    def rules(self) -> t.Tuple[Rule, ...]:
        rules_limit = config["adaptation"]["rules_limit"]
        slice = len(self._rules) if rules_limit == 0 else rules_limit

        return self._rules[:slice]

    @property
    def benchmark_rules(self) -> t.Tuple[Rule, ...]:
        return self._rules


@dataclass(frozen=True)
class WeightedScore:
    concept: Concept
    score: float
    weight: float

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            "concept": str(self.concept),
            "score": self.score,  # (1 - self.score) if negative else self.score,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class Evaluation:
    tp: t.List[WeightedScore]
    tn: t.Set[Concept]
    fp: t.List[WeightedScore]
    fn: t.List[WeightedScore]
    tp_score: float
    fn_score: float
    fp_score: float

    def to_dict(self, compact: bool = False) -> t.Dict[str, t.Any]:
        mapping = {
            "score": self.score,
            "tp_score": self.tp_score,
            "fn_score": self.fn_score,
            "fp_score": self.fp_score,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1": self.f1,
        }

        if not compact:
            mapping.update(
                {
                    "true_positives": convert.list_dict(self.tp),
                    "false_positives": convert.list_dict(self.fp),
                    "false_negatives": convert.list_dict(self.fn),
                }
            )

        return mapping

    @property
    def score(self) -> float:
        return (1 / 3) * (2 + self.tp_score - self.fn_score - self.fp_score)

    @property
    def precision(self) -> t.Optional[float]:
        den = len(self.tp) + len(self.fp)

        if den > 0:
            return len(self.tp) / den

        return None

    @property
    def recall(self) -> t.Optional[float]:
        den = len(self.tp) + len(self.fn)

        if den > 0:
            return len(self.tp) / den

        return None

    @property
    def f1(self) -> t.Optional[float]:
        den = 2 * len(self.tp) + len(self.fp) + len(self.fn)

        if den > 0:
            return (2 * len(self.tp)) / (den)

        return None

    @property
    def accuracy(self) -> t.Optional[float]:
        den = len(self.tp) + len(self.tn) + len(self.fp) + len(self.fn)

        if den > 0:
            return (len(self.tp) + len(self.tn)) / den

        return None

    @property
    def balanced_accuracy(self) -> t.Optional[float]:
        tpr = None
        tnr = None

        tpr_den = len(self.tp) + len(self.fn)
        tnr_den = len(self.tn) + len(self.fp)

        if tpr_den > 0:
            tpr = len(self.tp) / tpr_den

        if tnr_den > 0:
            tnr = len(self.tn) / tnr_den

        if tnr and tpr:
            return (tpr + tnr) / 2

        return None

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


def pos2spacy(pos: t.Optional[POS]) -> t.List[t.Optional[str]]:
    if pos == POS.NOUN:
        return ["NOUN"]  # "PROPN"
    elif pos == POS.VERB:
        return ["VERB"]
    elif pos == POS.ADJECTIVE:
        return ["ADJ"]
    elif pos == POS.ADVERB:
        return ["ADV"]

    return [None]
