import json
from gibberish import Gibberish
from recap_argument_graph_adaptation.model.adaptation import Concept
import neo4j
from .config import config
from . import graph
from ..util import conceptnet
import typing as t


class Database:
    _driver: neo4j.Neo4jDriver
    lang: str

    def __init__(self):
        driver = neo4j.GraphDatabase.driver(
            config["neo4j"]["url"],
            auth=(
                config["neo4j"]["username"],
                config["neo4j"]["password"],
            ),
            encrypted=False,
        )

        if isinstance(driver, neo4j.Neo4jDriver):
            self._driver = driver
            self.lang = config["nlp"]["lang"]

    # NODE

    def nodes(self, name: str, pos: graph.POS) -> t.Tuple[graph.Node, ...]:
        with self._driver.session() as session:
            return session.read_transaction(self._nodes, name, pos, self.lang)

    @staticmethod
    def _nodes(
        tx: neo4j.Session,
        name: str,
        pos: graph.POS,
        lang: str,
    ) -> t.Tuple[graph.Node, ...]:
        nodes = []
        query = "MATCH (n:Concept {name: $name, pos: $pos, language: $lang}) RETURN n"

        # We follow all available 'FormOf' relations to simplify path construction
        if config["conceptnet"]["nodes"]["root_form"]:
            query = (
                "MATCH p=((n:Concept {name: $name, pos: $pos, language: $lang})"
                "-[:FormOf*0..]->"
                "(m:Concept {language: $lang})) "
                "RETURN p ORDER BY length(p) DESC LIMIT 1"
            )

        # First, run the query with the given POS.
        # If no concept is found, retry the query without specifying a POS.
        # TODO: For some concepts, the node with a pos comes after the one without.
        # Example: health effects/noun is not in conceptnet, so effect is used.
        # Here, effect comes before effect/noun
        pos_tags = [pos]

        if pos != graph.POS.OTHER:
            pos_tags.append(graph.POS.OTHER)

        for pos_tag in pos_tags:
            record = tx.run(
                query,
                name=conceptnet.concept_name(name, lang),
                pos=pos_tag.value,
                lang=lang,
            ).single()

            if record:
                found_path = graph.Path.from_neo4j(record.value())
                found_nodes = reversed(found_path.nodes)

                # It can be the case that the concept name/pos exists, but ConceptNet returns name/other (due to missing relations).
                # Example: school uniform/other is returned for school uniforms/noun, but we want school uniform/noun
                # In the following, we will handle this scenario.
                end_node = found_path.end_node

                if end_node.pos != pos and (
                    best_node := Database._node(tx, end_node.name, pos, lang)
                ):
                    nodes.append(best_node)

                for found_node in found_nodes:
                    if found_node not in nodes:
                        nodes.append(found_node)

        return tuple(nodes)

    @staticmethod
    def _node(
        tx: neo4j.Session,
        name: str,
        pos: graph.POS,
        lang: str,
    ) -> t.Optional[graph.Node]:
        query = "MATCH (n:Concept {name: $name, pos: $pos, language: $lang}) RETURN n"

        record = tx.run(
            query,
            name=conceptnet.concept_name(name, lang),
            pos=pos.value,
            lang=lang,
        ).single()

        if record:
            return graph.Node.from_neo4j(record.value())

        return None

    # SHORTEST PATH

    def shortest_path(
        self, start_nodes: t.Sequence[graph.Node], end_nodes: t.Sequence[graph.Node]
    ) -> t.Optional[graph.Path]:
        relation_types = config["conceptnet"]["relations"]["generalization_types"]
        max_relations = config["conceptnet"]["paths"]["max_length"]

        with self._driver.session() as session:
            return session.read_transaction(
                self._shortest_path,
                start_nodes,
                end_nodes,
                relation_types,
                max_relations,
            )

    @staticmethod
    def _shortest_path(
        tx: neo4j.Session,
        start_nodes: t.Sequence[graph.Node],
        end_nodes: t.Sequence[graph.Node],
        relation_types: t.Collection[str],
        max_relations: int,
    ) -> t.Optional[graph.Path]:
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
                return graph.Path.from_neo4j(record.value())

        return None

    # ALL SHORTEST PATHS

    def all_shortest_paths(
        self, start_nodes: t.Sequence[graph.Node], end_nodes: t.Sequence[graph.Node]
    ) -> t.Optional[t.List[graph.Path]]:
        relation_types = config["conceptnet"]["relations"]["generalization_types"]
        max_relations = config["conceptnet"]["paths"]["max_length"]

        with self._driver.session() as session:
            return session.read_transaction(
                self._all_shortest_paths,
                start_nodes,
                end_nodes,
                relation_types,
                max_relations,
            )

    @staticmethod
    def _all_shortest_paths(
        tx: neo4j.Session,
        start_nodes: t.Sequence[graph.Node],
        end_nodes: t.Sequence[graph.Node],
        relation_types: t.Collection[str],
        max_relations: int,
    ) -> t.Optional[t.List[graph.Path]]:
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
                return [graph.Path.from_neo4j(record) for record in records]
        return None

    # EXPAND NODE

    def expand_nodes(
        self,
        nodes: t.Sequence[graph.Node],
        relation_types: t.Optional[t.Collection[str]] = None,
    ) -> t.Optional[t.List[graph.Path]]:
        if not relation_types:
            relation_types = config["conceptnet"]["relations"]["generalization_types"]

        with self._driver.session() as session:
            return session.read_transaction(
                self._expand_nodes, nodes, relation_types, self.lang
            )

    # TODO: Currently, only the first node that returns a result is queried.
    # Could be updated if results are not satisfactory.
    @staticmethod
    def _expand_nodes(
        tx: neo4j.Session,
        nodes: t.Sequence[graph.Node],
        relation_types: t.Collection[str],
        lang: str,
    ) -> t.Optional[t.List[graph.Path]]:
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
                return [graph.Path.from_neo4j(record) for record in records]

        return None

    # ADAPT PATH

    def adapt_paths(
        self,
        reference_paths: t.Sequence[graph.Path],
        start_nodes: t.Sequence[graph.Node],
    ) -> t.List[graph.Path]:
        with self._driver.session() as session:
            return session.read_transaction(
                self._adapt_paths, reference_paths, start_nodes, self.lang
            )

    @staticmethod
    def _adapt_paths(
        tx: neo4j.Session,
        reference_paths: t.Sequence[graph.Path],
        start_nodes: t.Sequence[graph.Node],
        lang: str,
        relax_relationship_types: bool = False,
    ) -> t.List[graph.Path]:
        adapted_paths = []

        for reference_path in reference_paths:
            query = "MATCH p=((n:Concept)"

            for rel in reference_path.relationships:
                rel_type = rel.type

                if relax_relationship_types:
                    rel_type = _include_relations(
                        config["conceptnet"]["relations"]["generalization_types"]
                    )

                query += f"-[:{rel_type}]{_arrow()}" "(:Concept {language: $lang})"

            query += ") WHERE id(n)=$start_id RETURN p"

            for node in start_nodes:
                records = tx.run(query, start_id=node.id, lang=lang).value()

                if records:
                    adapted_paths += [
                        graph.Path.from_neo4j(record) for record in records
                    ]

        if not adapted_paths and not relax_relationship_types:
            adapted_paths = Database._adapt_paths(
                tx, reference_paths, start_nodes, lang, relax_relationship_types
            )

        return adapted_paths

    # DISTANCE

    def distance(
        self, nodes1: t.Sequence[graph.Node], nodes2: t.Sequence[graph.Node]
    ) -> int:
        max_relations = 200
        relation_types = ["RelatedTo"]

        with self._driver.session() as session:
            return session.read_transaction(
                self._distance, nodes1, nodes2, relation_types, max_relations
            )

    @staticmethod
    def _distance(
        tx: neo4j.Session,
        nodes1: t.Sequence[graph.Node],
        nodes2: t.Sequence[graph.Node],
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

        return 0


def _arrow() -> str:
    return "->" if config["conceptnet"]["relations"]["directed"] else "-"


def _include_relations(allowed_types: t.Collection[str]) -> str:
    if allowed_types:
        return ":" + "|".join(allowed_types)

    return ""


def _exclude_relations(forbidden_types: t.Collection[str]) -> str:
    allowed_types = [
        rel_type
        for rel_type in config["conceptnet"]["relations"]["all_types"]
        if rel_type not in forbidden_types
    ]

    return _include_relations(allowed_types)


def _node_ids(nodes: t.Iterable[graph.Node]) -> t.List[int]:
    return [node.id for node in nodes]


def _iterate_nodes(
    nodes1: t.Sequence[graph.Node], nodes2: t.Sequence[graph.Node]
) -> t.Iterator[t.Tuple[graph.Node, graph.Node]]:
    iterator = []

    for index1 in range(len(nodes1)):
        for index2 in range(len(nodes2)):
            iterator.append((index1, index2))

    iterator.sort(key=lambda x: (sum(x), abs(x[0] - x[1])))

    for entry in iterator:
        yield (nodes1[entry[0]], nodes2[entry[1]])
