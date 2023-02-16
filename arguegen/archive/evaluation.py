from __future__ import annotations

import itertools
import math
import statistics
import typing as t
from dataclasses import dataclass

from arguegen.controllers import convert
from arguegen.model import casebase


@dataclass(frozen=True)
class WeightedScore:
    concept: casebase.Concept
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
                    f"The given metric '{metric}' is unknown. Possible values are"
                    f" '{aggr_funcs.keys()}'."
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
    tn: t.Set[casebase.Concept]
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
        num_negative = value < 0

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
