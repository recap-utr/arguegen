from __future__ import annotations

import itertools
import typing as t
from collections import defaultdict
from dataclasses import dataclass

import neo4j
import neo4j.data
from recap_argument_graph_adaptation.model import casebase, conceptnet_helper, graph
from recap_argument_graph_adaptation.model.config import Config

config = Config.instance()
POS_OTHER = "other"


@dataclass(frozen=True)
class ConceptnetNode(graph.AbstractNode):
    id: int
    language: str
    source: str

    @classmethod
    def from_neo4j(cls, obj: neo4j.data.Node) -> ConceptnetNode:
        return cls(
            id=obj.id,
            name=obj["name"],  # type: ignore
            language=obj["language"],  # type: ignore
            pos=obj["pos"],  # type: ignore
            uri=obj["uri"],  # type: ignore
            source=obj["source"],  # type: ignore
        )

    def hypernym_distances(self) -> t.Dict[ConceptnetNode, int]:
        return Database().hypernym_distances(self)


@dataclass(frozen=True)
class ConceptnetRelationship(graph.AbstractRelationship):
    # start_node: Node
    # end_node: Node
    id: int
    weight: float
    source: str

    @classmethod
    def from_neo4j(cls, obj: neo4j.data.Relationship) -> ConceptnetRelationship:
        return cls(
            id=obj.id,
            type=obj.type,
            start_node=ConceptnetNode.from_neo4j(obj.start_node),  # type: ignore
            end_node=ConceptnetNode.from_neo4j(obj.end_node),  # type: ignore
            uri=obj["uri"],  # type: ignore
            weight=obj["weight"],  # type: ignore
            source=obj["source"],  # type: ignore
        )


@dataclass(frozen=True)
class ConceptnetPath(graph.AbstractPath):
    # nodes: t.Tuple[Node, ...]
    # relationships: t.Tuple[Relationship, ...]

    @classmethod
    def from_neo4j(cls, obj: neo4j.data.Path) -> ConceptnetPath:
        return cls(
            nodes=tuple(ConceptnetNode.from_neo4j(node) for node in obj.nodes),
            relationships=tuple(
                ConceptnetRelationship.from_neo4j(rel) for rel in obj.relationships
            ),
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
        pos: t.Optional[str],
        lang: str,
        relation_types: t.Iterable[str],
        max_relations: int = 50,
        only_end_nodes: bool = True,
        exclude_start_node: bool = False,
        only_longest_path: bool = True,
    ) -> t.Dict[ConceptnetNode, int]:
        nodes = defaultdict(list)
        rel_query = _include_relations(relation_types)
        start_index = 1 if exclude_start_node else 0
        limit = (
            1 if only_longest_path else max_relations
        )  # TODO: Check if max_relations sensible

        query = (
            "MATCH p=((n:Concept {name: $name, pos: $pos, language: $lang})"
            f"-[{rel_query}*{start_index}..{max_relations}]->"
            "(m:Concept {language: $lang})) "
            f"RETURN p ORDER BY length(p) DESC LIMIT {limit}"
        )

        pos_tags = [pos]

        if pos:
            pos_tags.append(POS_OTHER)  # type: ignore

        for pos_tag in pos_tags:
            records = tx.run(
                query,
                name=conceptnet_helper.concept_name(name, lang),
                pos=pos_tag,
                lang=lang,
            ).value()

            if records:
                for record in records:
                    found_path = ConceptnetPath.from_neo4j(record)

                    if len(found_path.nodes) > 0:
                        # It can be the case that the concept name/pos exists, but ConceptNet returns name/other (due to missing relations).
                        # Example: school_uniform/other is returned for school_uniforms/noun, but we want school_uniform/noun
                        # In the following, we will handle this scenario.
                        end_node = found_path.end_node
                        nodes[end_node].append(len(found_path.relationships))

                        if end_node.pos != pos and (
                            best_node := Database._node(tx, end_node.name, pos, lang)
                        ):
                            nodes[end_node].append(len(found_path.relationships))

                        if not only_end_nodes:
                            for i, found_node in enumerate(found_path.nodes):
                                nodes[found_node].append(i)

        return {node: min(indices) for node, indices in nodes.items()}

    def nodes(
        self, name: str, pos: t.Optional[casebase.POS]
    ) -> t.FrozenSet[ConceptnetNode]:
        if self.active:
            with self._driver.session() as session:
                return session.read_transaction(self._nodes, name, pos.value, self.lang)

        return frozenset()

    @staticmethod
    def _nodes(
        tx: neo4j.Session,
        name: str,
        pos: t.Optional[str],
        lang: str,
    ) -> t.FrozenSet[ConceptnetNode]:
        # We follow all available 'FormOf' relations to simplify path construction
        if config["conceptnet"]["node"]["root_form"]:
            return frozenset(
                Database._nodes_along_paths(tx, name, pos, lang, ["FormOf"])
            )

        elif node := Database._node(tx, name, pos, lang):
            return frozenset([node])

        return frozenset()

    @staticmethod
    def _node(
        tx: neo4j.Session,
        name: str,
        pos: t.Optional[str],
        lang: str,
    ) -> t.Optional[ConceptnetNode]:
        query = "MATCH (n:Concept {name: $name, pos: $pos, language: $lang}) RETURN n"

        record = tx.run(
            query,
            name=conceptnet_helper.concept_name(name, lang),
            pos=pos,
            lang=lang,
        ).single()

        if record:
            return ConceptnetNode.from_neo4j(record.value())

        return None

    # HYPERNYMS

    def hypernym_distances(self, node: ConceptnetNode) -> t.Dict[ConceptnetNode, int]:
        if self.active:
            with self._driver.session() as session:
                return session.read_transaction(
                    self._hypernym_distances, node, self.lang
                )

        return {}

    # Currently, this function might return the same node that was given as input.
    # To resolve this, we would need to fork the function _nodes_along_paths.
    # TODO: Rewrite such that the distances are returned.
    @staticmethod
    def _hypernym_distances(
        tx: neo4j.Session,
        node: ConceptnetNode,
        lang: str,
    ) -> t.Dict[ConceptnetNode, int]:
        return Database._nodes_along_paths(
            tx,
            node.name,
            node.pos,
            lang,
            config["conceptnet"]["relation"]["generalization_types"],
            max_relations=config["conceptnet"]["path"]["max_length"]["generalization"],
            only_end_nodes=False,
            exclude_start_node=True,
            only_longest_path=False,
        )

    def hypernym_paths(
        self,
        node: ConceptnetNode,
        relation_types: t.Optional[t.Collection[str]] = None,
    ) -> t.FrozenSet[ConceptnetPath]:
        if not relation_types:
            relation_types = config["conceptnet"]["relation"]["generalization_types"]

        if self.active:
            with self._driver.session() as session:
                return session.read_transaction(
                    self._hypernym_paths, node, relation_types, self.lang
                )

        return frozenset()

    # Currently, only the first node that returns a result is queried.
    # Could be updated if results are not satisfactory.
    @staticmethod
    def _hypernym_paths(
        tx: neo4j.Session,
        node: ConceptnetNode,
        relation_types: t.Collection[str],
        lang: str,
    ) -> t.FrozenSet[ConceptnetPath]:
        rel_query = _include_relations(relation_types)

        query = (
            "MATCH p=((n:Concept)"
            f"-[r{rel_query}]{_arrow()}"
            "(m:Concept {language: $lang})) "
            "WHERE id(n)=$node_id "
            "RETURN p"
        )

        records = tx.run(query, node_id=node.id, lang=lang).value()

        if records:
            return frozenset({ConceptnetPath.from_neo4j(record) for record in records})

        return frozenset()

    # ALL SHORTEST PATHS
    # TODO: Check for overlap between hypernyms and all_shortest_paths
    def all_shortest_paths(
        self,
        start_nodes: t.Iterable[ConceptnetNode],
        end_nodes: t.Iterable[ConceptnetNode],
    ) -> t.List[ConceptnetPath]:
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

        return []

    @staticmethod
    def _all_shortest_paths(
        tx: neo4j.Session,
        start_nodes: t.Iterable[ConceptnetNode],
        end_nodes: t.Iterable[ConceptnetNode],
        relation_types: t.Collection[str],
        max_relations: int,
    ) -> t.List[ConceptnetPath]:
        rel_query = _include_relations(relation_types)

        query = (
            "MATCH p = allShortestPaths((n:Concept)"
            f"-[{rel_query}*..{max_relations}]{_arrow()}"
            "(m:Concept)) "
            "WHERE id(n) = $start_id AND id(m) = $end_id "
            "RETURN p"
        )

        nodes_iter = itertools.product(start_nodes, end_nodes)

        for nodes_pair in nodes_iter:
            records = tx.run(
                query, start_id=nodes_pair[0].id, end_id=nodes_pair[1].id
            ).value()

            if records:
                return [ConceptnetPath.from_neo4j(record) for record in records]

        return []

    # ADAPT PATH

    def adapt_paths(
        self,
        reference_paths: t.Iterable[ConceptnetPath],
        start_nodes: t.Iterable[ConceptnetNode],
    ) -> t.List[ConceptnetPath]:
        if self.active:
            with self._driver.session() as session:
                return session.read_transaction(
                    self._adapt_paths, reference_paths, start_nodes, self.lang
                )

        return []

    @staticmethod
    def _adapt_paths(
        tx: neo4j.Session,
        reference_paths: t.Iterable[ConceptnetPath],
        start_nodes: t.Iterable[ConceptnetNode],
        lang: str,
        relax_relationship_types: bool = False,
    ) -> t.List[ConceptnetPath]:
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
                    adapted_paths += [
                        ConceptnetPath.from_neo4j(record) for record in records
                    ]

        if not adapted_paths and not relax_relationship_types:
            adapted_paths = Database._adapt_paths(
                tx, reference_paths, start_nodes, lang, relax_relationship_types
            )

        return adapted_paths

    # METRICS

    def metrics(
        self, nodes1: t.Iterable[ConceptnetNode], nodes2: t.Iterable[ConceptnetNode]
    ) -> t.Dict[str, t.Optional[float]]:
        max_relations = config["nlp"]["max_distance"]
        relation_types = ["RelatedTo"]

        if self.active:
            with self._driver.session() as session:
                return session.read_transaction(
                    self._metrics, nodes1, nodes2, relation_types, max_relations
                )

        return {"path_similarity": None}

    @staticmethod
    def _metrics(
        tx: neo4j.Session,
        nodes1: t.Iterable[ConceptnetNode],
        nodes2: t.Iterable[ConceptnetNode],
        relation_types: t.Collection[str],
        max_relations: int,
    ) -> t.Dict[str, t.Optional[float]]:
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

            return {"path_similarity": _dist2sim(shortest_length)}

        return {"path_similarity": None}


def _dist2sim(distance: t.Optional[float]) -> t.Optional[float]:
    if distance is not None:
        return 1 / (1 + distance)

    return None


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


def _node_ids(nodes: t.Iterable[ConceptnetNode]) -> t.List[int]:
    return [node.id for node in nodes]


# def _iterate_nodes(
#     nodes1: t.Sequence[ConceptnetNode], nodes2: t.Sequence[ConceptnetNode]
# ) -> t.Iterator[t.Tuple[ConceptnetNode, ConceptnetNode]]:
#     iterator = []

#     for index1 in range(len(nodes1)):
#         for index2 in range(len(nodes2)):
#             iterator.append((index1, index2))

#     iterator.sort(key=lambda x: (sum(x), abs(x[0] - x[1])))

#     for entry in iterator:
#         yield (nodes1[entry[0]], nodes2[entry[1]])
