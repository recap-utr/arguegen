from __future__ import annotations

import abc
import typing as t
from dataclasses import dataclass

from recap_argument_graph_adaptation.model import spacy
from recap_argument_graph_adaptation.model.config import Config

config = Config.instance()


@dataclass(frozen=True)
class AbstractNode(abc.ABC):
    name: str
    _lemmas: t.FrozenSet[str]
    pos: t.Optional[str]
    uri: str

    @property
    def lemmas(self) -> t.FrozenSet[str]:
        return self._lemmas or frozenset([self.name])

    def __str__(self):
        if self.pos:
            return f"{self.name}/{self.pos}"

        return self.name

    def __eq__(self, other: AbstractNode) -> bool:
        return self.uri == other.uri

    def __hash__(self) -> int:
        return hash((self.uri,))

    @abc.abstractmethod
    def hypernym_distances(
        self, comparison_vectors: t.Iterable[spacy.Vector], min_similarity: float
    ) -> t.Dict[AbstractNode, int]:
        pass

    @property
    def processed_name(self) -> str:
        return process_name(self.name)

    @property
    def processed_lemmas(self) -> t.FrozenSet[str]:
        return frozenset(process_name(lemma) for lemma in self.lemmas)


def process_name(name: str) -> str:
    return name.replace("_", " ").lower()


@dataclass(frozen=True)
class AbstractRelationship(abc.ABC):
    type: str
    start_node: AbstractNode
    end_node: AbstractNode

    @property
    def nodes(self) -> t.Tuple[AbstractNode, AbstractNode]:
        return (self.start_node, self.end_node)

    def __str__(self):
        return f"{self.start_node}-[{self.type}]->{self.end_node}"


@dataclass(frozen=True)
class AbstractPath(abc.ABC):
    nodes: t.Tuple[AbstractNode, ...]
    relationships: t.Tuple[AbstractRelationship, ...]

    @property
    def start_node(self) -> AbstractNode:
        return self.nodes[0]

    @property
    def end_node(self) -> AbstractNode:
        return self.nodes[-1]

    def __str__(self):
        out = f"{self.start_node}"

        if len(self.nodes) > 1:
            for node, rel in zip(self.nodes[1:], self.relationships):
                out += f"-[{rel.type}]->{node}"

        return out

    def __len__(self) -> int:
        return len(self.relationships)

    @classmethod
    def from_node(cls, obj: AbstractNode) -> AbstractPath:
        return cls(nodes=(obj,), relationships=tuple())

    @classmethod
    def merge(cls, *paths: AbstractPath) -> AbstractPath:
        nodes = paths[0].nodes
        relationships = paths[0].relationships

        for path in paths[1:]:
            nodes += path.nodes[1:]
            relationships += path.relationships

        return cls(
            nodes=nodes,
            relationships=relationships,
        )
