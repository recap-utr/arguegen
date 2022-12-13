from __future__ import annotations

import itertools
import logging
import math
import statistics
import typing as t
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import arguebuf as ag
import immutables
import wn.constants

from arguegen.config import config, tuning
from arguegen.controller import convert
from arguegen.model import nlp, wordnet

log = logging.getLogger(__name__)

global_metrics = {
    "concept_sem_sim",
    "nodes_path_sim",
    "nodes_sem_sim",
    "nodes_wup_sim",
}

extraction_adaptation_metrics = {
    "query_concept_sem_sim",
    "query_nodes_sem_sim",
}

metrics_per_stage = {
    "extraction": {
        *global_metrics,
        *extraction_adaptation_metrics,
        "adus_sem_sim",
        "query_adus_sem_sim",
        "major_claim_prox",
        "keyword_weight",
    },
    "adaptation": {
        *global_metrics,
        *extraction_adaptation_metrics,
        "hypernym_prox",
    },
    "evaluation": {*global_metrics},
}

metric_keys = {
    "adus_sem_sim",
    "concept_sem_sim",
    "hypernym_prox",
    "keyword_weight",
    "major_claim_prox",
    "nodes_path_sim",
    "nodes_sem_sim",
    "nodes_wup_sim",
    "query_adus_sem_sim",
    "query_concept_sem_sim",
    "query_nodes_sem_sim",
}

assert metric_keys == set(itertools.chain.from_iterable(metrics_per_stage.values()))
empty_metrics: t.Callable[[], t.Dict[str, t.Optional[float]]] = lambda: {
    key: None for key in metric_keys
}
best_metrics: t.Callable[[], t.Dict[str, t.Optional[float]]] = lambda: {
    key: 1.0 for key in metric_keys
}


class HashableNode(ag.Node):
    def __str__(self) -> str:
        return str(self.id)

    def __eq__(self, other: HashableNode) -> bool:
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


class HashableAtom(HashableNode, ag.AtomNode):
    pass


class HashableScheme(HashableNode, ag.SchemeNode):
    pass


@dataclass(frozen=True, eq=True)
class Concept:
    lemma: str
    form2pos: immutables.Map[str, t.Tuple[str, ...]]
    pos2form: immutables.Map[str, t.Tuple[str, ...]]
    pos: t.Optional[POS]
    atoms: t.FrozenSet[HashableAtom]
    synsets: t.FrozenSet[wordnet.Node] = field(compare=False)
    related_concepts: t.Mapping[Concept, float] = field(compare=False)
    user_query: UserQuery = field(compare=False)
    metrics: t.Dict[str, t.Optional[float]] = field(
        default_factory=empty_metrics, compare=False, repr=False
    )

    @property
    def forms(self) -> t.FrozenSet[str]:
        return frozenset(self.form2pos.keys())

    def __str__(self):
        code = self.code

        if self.atoms:
            code += f"/{set(atom.id for atom in self.atoms)}"

        return code

    def part_eq(self, other: Concept) -> bool:
        return self.pos == other.pos and self.atoms == other.atoms

    @property
    def code(self) -> str:
        out = f"{self.lemma}"

        if self.pos:
            out += f"/{self.pos.value}"

        return out

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
            "nodes": [str(node) for node in self.synsets],
            "score": self.score,
        }

    @classmethod
    def from_concept(
        cls,
        source: Concept,
        lemma=None,
        form2pos=None,
        pos2form=None,
        pos=None,
        atoms=None,
        nodes=None,
        related_concepts=None,
        user_query=None,
        metrics=None,
    ) -> Concept:
        return cls(
            lemma or source.lemma,
            form2pos or source.form2pos,
            pos2form or source.pos2form,
            pos or source.pos,
            atoms or source.atoms,
            nodes or source.synsets,
            related_concepts or source.related_concepts,
            user_query or source.user_query,
            metrics or source.metrics,
        )


def score(metrics: t.Dict[str, t.Optional[float]]) -> float:
    result = 0
    total_weight = 0

    for metric_name, metric_weight in tuning(config, "score").items():
        if (metric := metrics[metric_name]) is not None:
            result += metric * metric_weight
            total_weight += metric_weight

    # Normalize the result.
    if total_weight > 0:
        return result / total_weight

    return 0.0


def filter_concepts(
    concepts: t.Iterable[Concept], min_score: float, topn: t.Optional[int]
) -> t.Set[Concept]:
    sorted_concepts = sorted(concepts, key=lambda x: x.score, reverse=True)
    filtered_concepts = list(filter(lambda x: x.score > min_score, sorted_concepts))

    if topn and topn > 0:
        filtered_concepts = filtered_concepts[:topn]

    return set(filtered_concepts)


@dataclass(frozen=True, eq=True)
class Rule:
    source: Concept
    target: Concept

    def __str__(self) -> str:
        return f"({self.source})->({self.target})"


@dataclass(frozen=True, eq=True)
class UserQuery:
    text: str

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True, eq=True)
class Case:
    relative_path: Path
    user_query: UserQuery
    graph: ag.Graph
    rules: t.Tuple[Rule, ...]
    benchmark_rules: t.Tuple[Rule, ...]

    def __str__(self) -> str:
        return str(self.relative_path)


class POS(Enum):
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"


def spacy2pos(pos: t.Optional[str]) -> t.Optional[POS]:
    if pos is None:
        return None

    return {
        "NOUN": POS.NOUN,
        "PROPN": POS.NOUN,
        "VERB": POS.VERB,
        "ADJ": POS.ADJECTIVE,
        "ADV": POS.ADVERB,
    }[pos]


def wn2pos(pos: t.Optional[str]) -> t.Optional[POS]:
    if pos is None:
        return None

    return {
        wn.constants.NOUN: POS.NOUN,
        wn.constants.VERB: POS.VERB,
        wn.constants.ADJECTIVE: POS.ADJECTIVE,
        wn.constants.ADJECTIVE_SATELLITE: POS.ADJECTIVE,
        wn.constants.ADVERB: POS.ADVERB,
    }.get(pos)


def pos2wn(pos: t.Optional[POS]) -> t.List[t.Optional[str]]:
    if pos is None:
        return [None]

    mapping: dict[POS, list[t.Optional[str]]] = {
        POS.NOUN: [wn.constants.NOUN],
        POS.VERB: [wn.constants.VERB],
        POS.ADJECTIVE: [wn.constants.ADJECTIVE, wn.constants.ADJECTIVE_SATELLITE],
        POS.ADVERB: [wn.constants.ADVERB],
    }

    return mapping.get(pos, [None])


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
