from __future__ import annotations

import typing as t
from dataclasses import dataclass
from enum import Enum

import neo4j
import neo4j.data
from recap_argument_graph_adaptation.model import casebase, conceptnet_helper
from recap_argument_graph_adaptation.model.config import Config

config = Config.instance()


class Language(Enum):
    EN = "en"
    DE = "de"


class Source(Enum):
    CONCEPTNET = "conceptnet"
    RECAP = "recap"


@dataclass(frozen=True)
class Node:
    id: int
    name: str
    pos: casebase.POS
    language: Language
    uri: str
    source: Source

    def __str__(self):
        if self.pos != casebase.POS.OTHER:
            return f"{self.name}/{self.pos.value}"

        return self.name

    @classmethod
    def from_neo4j(cls, obj: neo4j.data.Node) -> Node:
        return cls(
            id=obj.id,
            name=obj["name"],  # type: ignore
            language=Language(obj["language"]),  # type: ignore
            pos=casebase.POS(obj["pos"]),
            uri=obj["uri"],  # type: ignore
            source=Source(obj["source"]),  # type: ignore
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
    uri: str
    weight: float
    source: Source

    @property
    def nodes(self) -> t.Tuple[Node, Node]:
        return (self.start_node, self.end_node)

    def __str__(self):
        return f"{self.start_node}-[{self.type}]->{self.end_node}"

    @classmethod
    def from_neo4j(cls, obj: neo4j.data.Relationship) -> Relationship:
        return cls(
            id=obj.id,
            type=obj.type,
            start_node=Node.from_neo4j(obj.start_node),  # type: ignore
            end_node=Node.from_neo4j(obj.end_node),  # type: ignore
            uri=obj["uri"],  # type: ignore
            weight=obj["weight"],  # type: ignore
            source=Source(obj["source"]),  # type: ignore
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
    def from_neo4j(cls, obj: neo4j.data.Path) -> Path:
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


class Database:
    _driver: neo4j.Neo4jDriver
    lang: str
    active: bool

    def __init__(self):
        self.active = (
            True if config["adaptation"]["knowledge_graph"] == "conceptnet" else False
        )

        if self.active:
            driver = neo4j.GraphDatabase.driver(
                config["resources"]["conceptnet"]["url"],
                auth=(
                    config["resources"]["conceptnet"]["username"],
                    config["resources"]["conceptnet"]["password"],
                ),
                encrypted=False,
            )

            if isinstance(driver, neo4j.Neo4jDriver):
                self._driver = driver
                self.lang = config["nlp"]["lang"]

    # NODE

    @staticmethod
    def _nodes_along_paths(
        tx: neo4j.Session,
        name: str,
        pos: casebase.POS,
        lang: str,
        relation_types: t.Iterable[str],
        max_relations: int = 100,
        only_end_nodes: bool = True,
    ) -> t.Tuple[Node, ...]:
        nodes = []
        rel_query = _include_relations(relation_types)

        query = (
            "MATCH p=((n:Concept {name: $name, pos: $pos, language: $lang})"
            f"-[{rel_query}*0..{max_relations}]->"
            "(m:Concept {language: $lang})) "
            "RETURN p ORDER BY length(p) DESC LIMIT 1"
        )

        pos_tags = [pos]

        if pos != casebase.POS.OTHER:
            pos_tags.append(casebase.POS.OTHER)  # type: ignore

        for pos_tag in pos_tags:
            record = tx.run(
                query,
                name=conceptnet_helper.concept_name(name, lang),
                pos=pos_tag.value,
                lang=lang,
            ).single()

            if record:
                found_path = Path.from_neo4j(record.value())

                found_nodes = reversed(found_path.nodes)
                nodes_iter = iter(found_nodes)

                # It can be the case that the concept name/pos exists, but ConceptNet returns name/other (due to missing relations).
                # Example: school uniform/other is returned for school uniforms/noun, but we want school uniform/noun
                # In the following, we will handle this scenario.
                end_node = next(nodes_iter, None)

                if end_node:
                    nodes.append(end_node)

                    if end_node.pos != pos and (
                        best_node := Database._node(tx, end_node.name, pos, lang)
                    ):
                        nodes.append(best_node)

                    if not only_end_nodes:
                        for found_node in found_nodes:
                            if found_node not in nodes:
                                nodes.append(found_node)

        return tuple(nodes)

    def nodes(self, name: str, pos: casebase.POS) -> t.Tuple[Node, ...]:
        if self.active:
            with self._driver.session() as session:
                return session.read_transaction(self._nodes, name, pos, self.lang)

        return tuple()

    @staticmethod
    def _nodes(
        tx: neo4j.Session,
        name: str,
        pos: casebase.POS,
        lang: str,
    ) -> t.Tuple[Node, ...]:
        nodes = []

        # We follow all available 'FormOf' relations to simplify path construction
        if config["conceptnet"]["node"]["root_form"]:
            nodes = Database._nodes_along_paths(tx, name, pos, lang, ["FormOf"])

        elif node := Database._node(tx, name, pos, lang):
            nodes.append(node)

        return tuple(nodes)

    @staticmethod
    def _node(
        tx: neo4j.Session,
        name: str,
        pos: casebase.POS,
        lang: str,
    ) -> t.Optional[Node]:
        query = "MATCH (n:Concept {name: $name, pos: $pos, language: $lang}) RETURN n"

        record = tx.run(
            query,
            name=conceptnet_helper.concept_name(name, lang),
            pos=pos.value,
            lang=lang,
        ).single()

        if record:
            return Node.from_neo4j(record.value())

        return None

    # GENERALIZATION

    def generalizations(self, name: str, pos: casebase.POS) -> t.Tuple[Node, ...]:
        if self.active:
            with self._driver.session() as session:
                return session.read_transaction(
                    self._generalizations, name, pos, self.lang
                )

        return tuple()

    @staticmethod
    def _generalizations(
        tx: neo4j.Session,
        name: str,
        pos: casebase.POS,
        lang: str,
    ) -> t.Tuple[Node, ...]:
        relation_types = config["conceptnet"]["relation"]["generalization_types"] + [
            "FormOf"
        ]

        return Database._nodes_along_paths(
            tx,
            name,
            pos,
            lang,
            relation_types,
            max_relations=config["conceptnet"]["path"]["max_length"]["generalization"],
            only_end_nodes=False,
        )

    def nodes_generalizations(self, nodes: t.Sequence[Node]) -> t.Tuple[Node, ...]:
        if self.active:
            with self._driver.session() as session:
                return session.read_transaction(
                    self._nodes_generalizations, nodes, self.lang
                )

        return tuple()

    # Currently, this function might return the same node that was given as input.
    # To resolve this, we would need to fork the function _nodes_along_paths.
    @staticmethod
    def _nodes_generalizations(
        tx: neo4j.Session,
        nodes: t.Sequence[Node],
        lang: str,
    ) -> t.Tuple[Node, ...]:
        generalized_nodes = []
        generalization_types = config["conceptnet"]["relation"]["generalization_types"]

        for node in nodes:
            new_nodes = Database._nodes_along_paths(
                tx,
                node.name,
                node.pos,
                lang,
                generalization_types,
                max_relations=config["conceptnet"]["path"]["max_length"][
                    "generalization"
                ],
                only_end_nodes=False,
            )

            for new_node in new_nodes:
                if new_node not in generalized_nodes:
                    generalized_nodes.append(new_node)

        return tuple(generalized_nodes)

    # SHORTEST PATH

    def shortest_path(
        self, start_nodes: t.Sequence[Node], end_nodes: t.Sequence[Node]
    ) -> t.Optional[Path]:
        relation_types = config["conceptnet"]["relation"]["generalization_types"]
        max_relations = config["conceptnet"]["path"]["max_length"]["shortest_path"]

        if self.active:
            with self._driver.session() as session:
                return session.read_transaction(
                    self._shortest_path,
                    start_nodes,
                    end_nodes,
                    relation_types,
                    max_relations,
                )

        return None

    @staticmethod
    def _shortest_path(
        tx: neo4j.Session,
        start_nodes: t.Sequence[Node],
        end_nodes: t.Sequence[Node],
        relation_types: t.Collection[str],
        max_relations: int,
    ) -> t.Optional[Path]:
        rel_query = _include_relations(relation_types)

        query = (
            "MATCH p = shortestPath((n:Concept)"
            f"-[{rel_query}*..{max_relations}]{_arrow()}"
            "(m:Concept)) "
            "WHERE id(n) = $start_id AND id(m) = $end_id"
            "RETURN p"
        )

        nodes_iter = _iterate_nodes(start_nodes, end_nodes)

        for nodes_pair in nodes_iter:
            record = tx.run(
                query, start_id=nodes_pair[0].id, end_id=nodes_pair[1].id
            ).single()

            if record:
                return Path.from_neo4j(record.value())

        return None

    # ALL SHORTEST PATHS

    def all_shortest_paths(
        self, start_nodes: t.Sequence[Node], end_nodes: t.Sequence[Node]
    ) -> t.Optional[t.List[Path]]:
        relation_types = config["conceptnet"]["relation"]["generalization_types"]
        max_relations = config["conceptnet"]["path"]["max_length"]["shortest_path"]

        if self.active:
            with self._driver.session() as session:
                return session.read_transaction(
                    self._all_shortest_paths,
                    start_nodes,
                    end_nodes,
                    relation_types,
                    max_relations,
                )

        return None

    @staticmethod
    def _all_shortest_paths(
        tx: neo4j.Session,
        start_nodes: t.Sequence[Node],
        end_nodes: t.Sequence[Node],
        relation_types: t.Collection[str],
        max_relations: int,
    ) -> t.Optional[t.List[Path]]:
        rel_query = _include_relations(relation_types)

        query = (
            "MATCH p = allShortestPaths((n:Concept)"
            f"-[{rel_query}*..{max_relations}]{_arrow()}"
            "(m:Concept)) "
            "WHERE id(n) = $start_id AND id(m) = $end_id "
            "RETURN p"
        )

        nodes_iter = _iterate_nodes(start_nodes, end_nodes)

        for nodes_pair in nodes_iter:
            records = tx.run(
                query, start_id=nodes_pair[0].id, end_id=nodes_pair[1].id
            ).value()

            if records:
                return [Path.from_neo4j(record) for record in records]
        return None

    # EXPAND NODE

    def expand_nodes(
        self,
        nodes: t.Sequence[Node],
        relation_types: t.Optional[t.Collection[str]] = None,
    ) -> t.Optional[t.List[Path]]:
        if not relation_types:
            relation_types = config["conceptnet"]["relation"]["generalization_types"]

        if self.active:
            with self._driver.session() as session:
                return session.read_transaction(
                    self._expand_nodes, nodes, relation_types, self.lang
                )

        return None

    # Currently, only the first node that returns a result is queried.
    # Could be updated if results are not satisfactory.
    @staticmethod
    def _expand_nodes(
        tx: neo4j.Session,
        nodes: t.Sequence[Node],
        relation_types: t.Collection[str],
        lang: str,
    ) -> t.Optional[t.List[Path]]:
        rel_query = _include_relations(relation_types)

        query = (
            "MATCH p=((n:Concept)"
            f"-[r{rel_query}]{_arrow()}"
            "(m:Concept {language: $lang})) "
            "WHERE id(n)=$node_id "
            "RETURN p"
        )

        for node in nodes:
            records = tx.run(query, node_id=node.id, lang=lang).value()

            if records:
                return [Path.from_neo4j(record) for record in records]

        return None

    # ADAPT PATH

    def adapt_paths(
        self,
        reference_paths: t.Sequence[Path],
        start_nodes: t.Sequence[Node],
    ) -> t.List[Path]:
        if self.active:
            with self._driver.session() as session:
                return session.read_transaction(
                    self._adapt_paths, reference_paths, start_nodes, self.lang
                )

        return []

    @staticmethod
    def _adapt_paths(
        tx: neo4j.Session,
        reference_paths: t.Sequence[Path],
        start_nodes: t.Sequence[Node],
        lang: str,
        relax_relationship_types: bool = False,
    ) -> t.List[Path]:
        adapted_paths = []

        for reference_path in reference_paths:
            query = "MATCH p=((n:Concept)"

            for rel in reference_path.relationships:
                rel_type = rel.type

                if relax_relationship_types:
                    rel_type = _include_relations(
                        config["conceptnet"]["relation"]["generalization_types"]
                    )

                query += f"-[:{rel_type}]{_arrow()}" "(:Concept {language: $lang})"

            query += ") WHERE id(n)=$start_id RETURN p"

            for node in start_nodes:
                records = tx.run(query, start_id=node.id, lang=lang).value()

                if records:
                    adapted_paths += [Path.from_neo4j(record) for record in records]

        if not adapted_paths and not relax_relationship_types:
            adapted_paths = Database._adapt_paths(
                tx, reference_paths, start_nodes, lang, relax_relationship_types
            )

        return adapted_paths

    # DISTANCE

    def distance(
        self, nodes1: t.Sequence[Node], nodes2: t.Sequence[Node]
    ) -> t.Optional[int]:
        max_relations = config["nlp"]["max_distance"]
        relation_types = ["RelatedTo"]

        if self.active:
            with self._driver.session() as session:
                return session.read_transaction(
                    self._distance, nodes1, nodes2, relation_types, max_relations
                )

        return None

    @staticmethod
    def _distance(
        tx: neo4j.Session,
        nodes1: t.Sequence[Node],
        nodes2: t.Sequence[Node],
        relation_types: t.Collection[str],
        max_relations: int,
    ) -> int:
        # If there is any node equal in both sequences, we return a distance of 0.
        # In this case, the shortest path algorithm does not work.
        if set(nodes1).isdisjoint(nodes2):
            rel_query = _exclude_relations(relation_types)
            shortest_length = max_relations

            query = (
                "MATCH p = shortestPath((n:Concept)"
                f"-[r{rel_query}*..{max_relations}]-"
                "(m:Concept)) "
                f"WHERE id(n) IN $ids1 AND id(m) IN $ids2 "
                "RETURN LENGTH(p)"
            )

            record = tx.run(
                query,
                ids1=_node_ids(nodes1),
                ids2=_node_ids(nodes2),
            ).value()

            if record:
                shortest_length = min(record)

            return shortest_length

        return config["nlp"]["max_distance"]


def _arrow() -> str:
    return "->" if config["conceptnet"]["relation"]["directed"] else "-"


def _include_relations(allowed_types: t.Iterable[str]) -> str:
    if allowed_types:
        return ":" + "|".join(allowed_types)

    return ""


def _exclude_relations(forbidden_types: t.Iterable[str]) -> str:
    allowed_types = [
        rel_type
        for rel_type in config["conceptnet"]["relation"]["all_types"]
        if rel_type not in forbidden_types
    ]

    return _include_relations(allowed_types)


def _node_ids(nodes: t.Iterable[Node]) -> t.List[int]:
    return [node.id for node in nodes]


def _iterate_nodes(
    nodes1: t.Sequence[Node], nodes2: t.Sequence[Node]
) -> t.Iterator[t.Tuple[Node, Node]]:
    iterator = []

    for index1 in range(len(nodes1)):
        for index2 in range(len(nodes2)):
            iterator.append((index1, index2))

    iterator.sort(key=lambda x: (sum(x), abs(x[0] - x[1])))

    for entry in iterator:
        yield (nodes1[entry[0]], nodes2[entry[1]])
