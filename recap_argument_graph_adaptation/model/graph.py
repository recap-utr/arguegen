from __future__ import annotations

import typing as t
from dataclasses import dataclass

import neo4j


@dataclass(frozen=True)
class Node:
    id: int
    name: str
    language: str
    source: str

    def __str__(self):
        return f"({self.name})"

    @classmethod
    def from_neo4j(cls, obj: neo4j.Node) -> Node:
        return cls(
            id=obj.id, name=obj["name"], language=obj["language"], source=obj["source"]
        )

    @property
    def processed_name(self):
        return self.name.replace("_", " ")


@dataclass(frozen=True)
class Relationship:
    id: int
    type: str
    start_node: Node
    end_node: Node
    weight: float
    source: str

    @property
    def nodes(self) -> t.Tuple[Node, Node]:
        return (self.start_node, self.end_node)

    def __str__(self):
        return f"{self.start_node}-[{self.type}]->{self.end_node}"

    @classmethod
    def from_neo4j(cls, obj: neo4j.Relationship) -> Relationship:
        # TODO: The node objects in a relationship are not identical to the ones in a path object.
        return cls(
            id=obj.id,
            type=obj.type,
            start_node=Node.from_neo4j(obj.start_node),
            end_node=Node.from_neo4j(obj.end_node),
            weight=obj["weight"],
            source=obj["source"],
        )


@dataclass(frozen=True)
class Path:
    nodes: t.Tuple[Node, ...]
    relationships: t.Tuple[Relationship, ...]

    @property
    def start_node(self) -> Node:
        return self.nodes[0]

    @property
    def end_node(self) -> Node:
        return self.nodes[-1]

    def __str__(self):
        out = f"{self.start_node}"

        if len(self.nodes) > 1:
            for node, rel in zip(self.nodes[1:], self.relationships):
                out += f"-[{rel.type}]->{node}"

        return out

    @classmethod
    def from_neo4j(cls, obj: neo4j.Path) -> Path:
        return cls(
            nodes=tuple(Node.from_neo4j(node) for node in obj.nodes),
            relationships=tuple(
                Relationship.from_neo4j(rel) for rel in obj.relationships
            ),
        )

    @classmethod
    def from_node(cls, obj: Node) -> Path:
        return cls(nodes=(obj,), relationships=tuple())

    @classmethod
    def merge(cls, *paths: Path) -> Path:
        nodes = paths[0].nodes
        relationships = paths[0].relationships

        for path in paths[1:]:
            nodes += path.nodes[1:]
            relationships += path.relationships

        return cls(
            nodes=nodes,
            relationships=relationships,
        )
