from __future__ import annotations

import itertools
import logging
import math
import statistics
import typing as t
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Value
from pathlib import Path

import immutables
import recap_argument_graph as ag
from recap_argument_graph_adaptation.controller import convert
from recap_argument_graph_adaptation.model import graph, spacy
from recap_argument_graph_adaptation.model.config import Config

log = logging.getLogger(__name__)
config = Config.instance()

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


@dataclass(frozen=True, eq=True)
class Concept:
    name: str
    vector: spacy.Vector = field(compare=False, repr=False)
    form2pos: immutables.Map[str, t.Tuple[str, ...]]
    pos2form: immutables.Map[str, t.Tuple[str, ...]]
    pos: t.Optional[POS]
    inodes: t.FrozenSet[ArgumentNode]
    nodes: t.FrozenSet[graph.AbstractNode] = field(compare=False)
    related_concepts: t.Mapping[Concept, float] = field(compare=False)
    user_query: UserQuery = field(compare=False)
    metrics: t.Dict[str, t.Optional[float]] = field(
        default_factory=empty_metrics, compare=False, repr=False
    )

    @property
    def forms(self) -> t.Tuple[str, ...]:
        return self.form2pos.keys()

    def __str__(self):
        code = self.code

        if self.inodes:
            code += f"/{set(inode.key for inode in self.inodes)}"

        return code

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
        cls,
        source: Concept,
        name=None,
        vector=None,
        form2pos=None,
        pos2form=None,
        pos=None,
        inodes=None,
        nodes=None,
        related_concepts=None,
        user_query=None,
        metrics=None,
    ) -> Concept:
        if vector is None:
            vector = source.vector

        return cls(
            name or source.name,
            vector,
            form2pos or source.form2pos,
            pos2form or source.pos2form,
            pos or source.pos,
            inodes or source.inodes,
            nodes or source.nodes,
            related_concepts or source.related_concepts,
            user_query or source.user_query,
            metrics or source.metrics,
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
    vector: spacy.Vector = field(repr=False, compare=False)

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True, eq=True)
class Case:
    relative_path: Path
    user_query: UserQuery
    graph: ag.Graph
    _rules: t.Tuple[Rule, ...]

    def __str__(self) -> str:
        return str(self.relative_path)

    @property
    def rules(self) -> t.Tuple[Rule, ...]:
        # slice = config.tuning("global", "rule_limit")
        rules_limit = config.tuning("global", "rule_limit")
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


def aggregate_eval(
    objects: t.Sequence[t.Union[Evaluation, EvaluationTuple]],
    metric: t.Optional[str] = None,
) -> t.Dict[str, t.Any]:
    if all(isinstance(obj, Evaluation) for obj in objects):
        aggr_funcs = {
            "mean": statistics.mean,
            "min": min,
            "max": max,
        }

        if len(objects) == 0:
            return {}
        elif len(objects) == 1:
            return objects[0].to_dict(compact=True)
        elif metric:
            if func := aggr_funcs[metric]:
                out = {}
                latex_elements = (
                    Evaluation._latex_format(
                        (func(getattr(object, key) or 0.0 for object in objects))
                    )
                    for key in Evaluation._latex_keys()
                )

                out["latex"] = " & ".join(itertools.chain([metric], latex_elements))

                out.update(
                    {
                        key: func(getattr(object, key) or 0.0 for object in objects)
                        for key in Evaluation.keys(compact=True)
                        if key != "latex"
                    }
                )

                return out
            else:
                raise ValueError(
                    f"The given metric '{metric}' is unknown. Possible values are '{aggr_funcs.keys()}'."
                )

        return {
            key: aggregate_eval(objects, key)
            for key in aggr_funcs.keys()
            if key in config["export"]["aggr_funcs"]
        }

    elif all(isinstance(obj, EvaluationTuple) for obj in objects):
        return {
            key: aggregate_eval([getattr(obj, key) for obj in objects])
            for key in export_keys
        }

    raise ValueError("All evaluation objects must have the same type.")


export_keys = (
    ["synthesis", "deliberation"] if config["export"]["deliberation"] else ["synthesis"]
)


@dataclass(frozen=True)
class EvaluationTuple:
    synthesis: Evaluation
    deliberation: Evaluation

    def to_dict(self, compact: bool = False) -> t.Dict[str, t.Any]:
        return {key: getattr(self, key).to_dict(compact) for key in export_keys}


@dataclass(frozen=True)
class Evaluation:
    duration: float
    tp: t.List[WeightedScore]
    tn: t.Set[Concept]
    fp: t.List[WeightedScore]
    fn: t.List[WeightedScore]
    tp_score: float
    fn_score: float
    fp_score: float
    # baseline_tp_score: float
    retrieved_sim: t.Optional[float]
    adapted_sim: t.Optional[float]

    @staticmethod
    def keys(compact: bool = False) -> t.List[str]:
        k = [
            "latex",
            "duration",
            "score",
            "tp_score",
            "fn_score",
            "fp_score",
            # "baseline_tp_score",
            # "baseline_tp_score_improvement",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "balanced_accuracy",
            "error_rate",
            "retrieved_sim",
            "adapted_sim",
            "sim_improvement",
        ]

        if not compact:
            k.extend(["true_positives", "false_positives", "false_negatives"])

        return k

    def to_dict(self, compact: bool = False) -> t.Dict[str, t.Any]:
        return {key: getattr(self, key) for key in Evaluation.keys(compact)}

    @property
    def latex(self) -> str:
        return " & ".join(
            Evaluation._latex_format(getattr(self, key) or 0.0)
            for key in Evaluation._latex_keys()
        )

    @staticmethod
    def _latex_format(value: float) -> str:
        prefix = str(abs(int(value)))
        num_negative = True if value < 0 else False

        if value < 1 and value > -1:
            num_digits = 3
        elif len(prefix) > 3:
            num_digits = 0
        else:
            num_digits = 3 - len(prefix)

        pattern = f"{{:.{num_digits}f}}"
        num_formatted = pattern.format(value).lstrip("-0")

        if num_negative:
            num_formatted += "-"

        return f"${num_formatted}$"

    @staticmethod
    def _latex_keys() -> t.List[str]:
        return [
            "duration",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "tp_score",
            "fn_score",
            "fp_score",
            "score",
            # "baseline_tp_score_improvement",
        ]

    @property
    def score(self) -> float:
        return (1 / 3) * (2 + self.tp_score - self.fn_score - self.fp_score)

    # @property
    # def baseline_score(self) -> float:
    #     return (1 / 3) * (2 + self.baseline_tp_score - self.fn_score)

    # @property
    # def baseline_tp_score_improvement(self) -> float:
    #     if self.baseline_tp_score > 0:
    #         return (self.tp_score / self.baseline_tp_score) - 1

    #     return 0.0

    # @property
    # def baseline_score_improvement(self) -> float:
    #     if self.baseline_score > 0:
    #         return (self.score / self.baseline_score) - 1

    #     return 0.0

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

    def f_score(self, beta: float) -> t.Optional[float]:
        prec = self.precision
        rec = self.recall

        if prec is not None and rec is not None:
            num = (1 + pow(beta, 2)) * prec * rec
            den = pow(beta, 2) * prec + rec

            if den > 0:
                return num / den

            return None

        return None

    @property
    def f1(self) -> t.Optional[float]:
        return self.f_score(1)

    @property
    def accuracy(self) -> t.Optional[float]:
        den = len(self.tp) + len(self.tn) + len(self.fp) + len(self.fn)

        if den > 0:
            return (len(self.tp) + len(self.tn)) / den

        return None

    @property
    def balanced_accuracy(self) -> t.Optional[float]:
        tpr = self.sensitivity
        tnr = self.specificity

        if tnr is not None and tpr is not None:
            return (tpr + tnr) / 2

        return None

    @property
    def error_rate(self) -> t.Optional[float]:
        den = len(self.tp) + len(self.tn) + len(self.fp) + len(self.fn)

        if den > 0:
            return (len(self.fp) + len(self.fn)) / den

        return None

    @property
    def sensitivity(self) -> t.Optional[float]:
        return self.recall

    @property
    def specificity(self) -> t.Optional[float]:
        den = len(self.tn) + len(self.fp)

        if den > 0:
            return len(self.tn) / den

        return None

    @property
    def geometric_mean(self) -> float:
        return math.sqrt(len(self.tp) * len(self.tn))

    @property
    def true_positives(self):
        return convert.list_dict(self.tp)

    @property
    def false_positives(self):
        return convert.list_dict(self.fp)

    @property
    def false_negatives(self):
        return convert.list_dict(self.fn)

    @property
    def sim_improvement(self):
        if self.adapted_sim and self.retrieved_sim:
            return (self.adapted_sim / self.retrieved_sim) - 1

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
