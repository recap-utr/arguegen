import typing as t
from dataclasses import dataclass
from .adaptation import Concept
from recap_argument_graph_adaptation.helper import convert


@dataclass
class Evaluation:
    score: float
    benchmark_and_computed: t.Set[Concept]
    only_benchmark: t.Set[Concept]
    only_computed: t.Set[Concept]

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            "score": self.score,
            "benchmark_and_computed": convert.list_str(self.benchmark_and_computed),
            "only_benchmark": convert.list_str(self.only_benchmark),
            "only_computed": convert.list_str(self.only_computed),
        }
