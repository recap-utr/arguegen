import typing as t
from dataclasses import dataclass

from recap_argument_graph_adaptation.helper import convert

from .adaptation import Concept


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
