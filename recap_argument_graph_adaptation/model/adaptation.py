from recap_argument_graph_adaptation.model.database import Database
from recap_argument_graph_adaptation.model import database
import recap_argument_graph as ag
from dataclasses import dataclass, field
from enum import Enum
import typing as t

log = logging.getLogger(__name__)


class Method(Enum):
    BETWEEN = "between"
    WITHIN = "within"


class Selector(Enum):
    DIFFERENCE = "difference"
    SIMILARITY = "similarity"


@dataclass
class Rule:
    source: str
    target: str
    source_conceptnet: str = field(init=False)
    target_conceptnet: str = field(init=False)

    def __post_init__(self) -> None:
        db = Database()

        try:
            self.source_conceptnet = db.node(self.source).name
            self.target_conceptnet = db.node(self.target).name
        except AttributeError:
            raise ValueError(
                "The concepts of the given rule cannot be found in ConceptNet."
            )

    def __getitem__(self, key: int) -> str:
        if key == 0:
            return self.source_conceptnet
        elif key == 1:
            return self.target_conceptnet
        else:
            raise ValueError("Wrong index provided.")


# TODO: An error is thrown if words of the rules do not exist in ConceptNet.


@dataclass(frozen=True)
class Case:
    name: str
    query: str
    graph: ag.Graph
    rules: t.List[Rule]
    benchmark_graph: ag.Graph
    benchmark_rules: t.List[Rule]


@dataclass(frozen=True)
class Concept:
    original_name: str
    conceptnet_name: str

    def __str__(self):
        return self.conceptnet_name
