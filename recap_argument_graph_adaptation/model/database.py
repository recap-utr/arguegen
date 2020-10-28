import json
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
        tx: neo4j.Session, name: str, pos: graph.POS, lang: str
    ) -> t.Tuple[graph.Node, ...]:
        nodes = []
        query = "MATCH (n:Concept {name: $name, pos: $pos, language: $lang}) RETURN n"

        # We follow all available 'FormOf' relations to simplify path construction
        if config["conceptnet"]["nodes"]["root_form"]:
            query = (
                "MATCH p=((n:Concept {name: $name, pos: $pos, language: $lang})"
                "-[:FormOf*0..]->"
                "(m:Concept {language: $lang})) "
                "RETURN m ORDER BY length(p) DESC LIMIT 1"
            )

        # First, run the query with the given POS.
        # If no concept is found, retry the query without specifying a POS.
        for pos_tag in set([pos, graph.POS.OTHER]):
            record = tx.run(
                query,
                name=conceptnet.concept_name(name, lang),
                pos=pos_tag.value,
                lang=lang,
            ).single()

            if record:
                nodes.append(graph.Node.from_neo4j(record.value()))

        return tuple(nodes)

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
            records = tx.run(query, start_id=nodes_pair[0].id, end_id=nodes_pair[1].id)

            if records:
                return [graph.Path.from_neo4j(record.value()) for record in records]
        return None

    # EXPAND NODE

    def expand_nodes(
        self, nodes: t.Sequence[graph.Node], relation_types: t.Collection[str] = None
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
            records = tx.run(query, node_id=node.id, lang=lang)

            if records:
                return [graph.Path.from_neo4j(record.value()) for record in records]

        return None

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

    # TODO: The shortest path algorithm does not work when the start and end nodes are the same.
    # This can happen if you perform a shortestPath search after a cartesian product that might
    # have the same start and end nodes for some of the rows passed to shortestPath.
    # If you would rather not experience this exception, and can accept the possibility of
    # missing results for those rows, disable this in the Neo4j configuration by setting
    # `cypher.forbid_shortestpath_common_nodes` to false. If you cannot accept missing results,
    # and really want the shortestPath between two common nodes, then re-write the query using
    # a standard Cypher variable length pattern expression followed by ordering by path length
    # and limiting to one result.
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
            )

            if record:
                shortest_length = min(record.value())

            return shortest_length

        else:
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
