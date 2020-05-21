import neo4j
from .config import config
from . import graph
from ..util import conceptnet
import typing as t


class Database:
    _driver: neo4j.Driver
    lang: str

    def __init__(self):
        self._driver = neo4j.GraphDatabase.driver(
            config["neo4j"]["url"],
            auth=(config["neo4j"]["username"], config["neo4j"]["password"]),
            encrypted=False,
        )
        self.lang = config["nlp"]["lang"]

    # NODE

    def node(self, name: str) -> t.Optional[graph.Node]:
        with self._driver.session() as session:
            return session.read_transaction(self._node, name, self.lang)

    @staticmethod
    def _node(tx: neo4j.Session, name: str, lang: str) -> t.Optional[graph.Node]:
        result = tx.run(
            "MATCH (n:Concept {name: $name, language: $lang}) RETURN n",
            name=conceptnet.concept_name(name, lang),
            lang=lang,
        )

        record = result.single()

        return graph.Node.from_neo4j(record.value()) if record else None

    # SHORTEST PATH

    def shortest_path(self, start: str, end: str) -> t.Optional[graph.Path]:
        relation_types = config["neo4j"]["relation_types"]
        max_relations = config["neo4j"]["max_relations"]

        with self._driver.session() as session:
            return session.read_transaction(
                self._shortest_path,
                start,
                end,
                relation_types,
                max_relations,
                self.lang,
            )

    @staticmethod
    def _shortest_path(
        tx: neo4j.Session,
        start: str,
        end: str,
        relation_types: t.Collection[str],
        max_relations: int,
        lang: str,
    ) -> t.Optional[graph.Path]:
        rel_query = _aggregate_relations(relation_types)

        result = tx.run(
            "MATCH p = shortestPath((n:Concept {name: $start, language: $lang})"
            f"-[{rel_query}*..{max_relations}]->"
            "(m:Concept {name: $end, language: $lang})) RETURN p",
            start=conceptnet.concept_name(start, lang),
            end=conceptnet.concept_name(end, lang),
            lang=lang,
        )

        record = result.single()

        return graph.Path.from_neo4j(record.value()) if record else None

    # ALL SHORTEST PATHS

    def all_shortest_paths(
        self, start: str, end: str
    ) -> t.Optional[t.List[graph.Path]]:
        relation_types = config["neo4j"]["relation_types"]
        max_relations = config["neo4j"]["max_relations"]

        with self._driver.session() as session:
            return session.read_transaction(
                self._all_shortest_paths,
                start,
                end,
                relation_types,
                max_relations,
                self.lang,
            )

    @staticmethod
    def _all_shortest_paths(
        tx: neo4j.Session,
        start: str,
        end: str,
        relation_types: t.Collection[str],
        max_relations: int,
        lang: str,
    ) -> t.Optional[t.List[graph.Path]]:
        rel_query = _aggregate_relations(relation_types)

        result = tx.run(
            "MATCH p = allShortestPaths((n:Concept {name: $start, language: $lang})"
            f"-[{rel_query}*..{max_relations}]->"
            "(m:Concept {name: $end, language: $lang})) RETURN p",
            start=conceptnet.concept_name(start, lang),
            end=conceptnet.concept_name(end, lang),
            lang=lang,
        )

        if result:
            return [graph.Path.from_neo4j(record.value()) for record in result]

        return None

    # EXPAND NODE

    def expand_node(
        self, node: graph.Node, relation_types: t.Collection[str] = None
    ) -> t.Optional[t.List[graph.Path]]:
        if not relation_types:
            relation_types = config["neo4j"]["relation_types"]

        with self._driver.session() as session:
            return session.read_transaction(
                self._expand_node, node, relation_types, self.lang
            )

    @staticmethod
    def _expand_node(
        tx: neo4j.Session,
        node: graph.Node,
        relation_types: t.Collection[str],
        lang: str,
    ) -> t.Optional[t.List[graph.Path]]:
        rel_query = _aggregate_relations(relation_types)
        result = tx.run(
            f"MATCH p=(n:Concept {{language: $lang}})-[r{rel_query}]->(m:Concept {{language: $lang}})"
            f"WHERE id(n)={node.id} RETURN p",
            lang=lang,
        )

        if result:
            return [graph.Path.from_neo4j(record.value()) for record in result]

        return None


def _aggregate_relations(relation_types: t.Collection[str]) -> str:
    if relation_types:
        return ":" + "|".join(relation_types)

    return ""
