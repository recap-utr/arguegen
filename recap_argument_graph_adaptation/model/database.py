import neo4j
from .config import config
from . import conceptnet, graph
from gibberish import Gibberish
import typing as t


class Database:
    _driver: neo4j.Driver
    lang: str

    def __init__(self):
        self._driver = neo4j.GraphDatabase.driver(
            config["neo4j"]["url"], auth=None, encrypted=False
        )  # auth=("username", "password")
        self.lang = config["neo4j"]["lang"]

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
        ).single()

        return graph.Node.from_neo4j(result.value()) if result else None

    # SHORTEST PATH

    def shortest_path(self, start: str, end: str) -> t.Optional[graph.Path]:
        with self._driver.session() as session:
            return session.read_transaction(self._shortest_path, start, end, self.lang)

    @staticmethod
    def _shortest_path(
        tx: neo4j.Session, start: str, end: str, lang: str
    ) -> t.Optional[graph.Path]:
        result = tx.run(
            "MATCH p = shortestPath((n:Concept {name: $start, language: $lang})-[*..10]-"
            "(m:Concept {name: $end, language: $lang})) RETURN p",
            start=conceptnet.concept_name(start, lang),
            end=conceptnet.concept_name(end, lang),
            # max_relations=max_relations,
            lang=lang,
        ).single()

        return graph.Path.from_neo4j(result.value()) if result else None

        # ALL SHORTEST PATHS

    def all_shortest_paths(
        self, start: str, end: str
    ) -> t.Optional[t.List[graph.Path]]:
        with self._driver.session() as session:
            return session.read_transaction(
                self._all_shortest_paths, start, end, self.lang
            )

    @staticmethod
    def _all_shortest_paths(
        tx: neo4j.Session, start: str, end: str, lang: str
    ) -> t.Optional[t.List[graph.Path]]:
        result = tx.run(
            "MATCH p = allShortestPaths((n:Concept {name: $start, language: $lang})-[*..10]-"
            "(m:Concept {name: $end, language: $lang})) RETURN p",
            start=conceptnet.concept_name(start, lang),
            end=conceptnet.concept_name(end, lang),
            # max_relations=max_relations,
            lang=lang,
        )

        if result:
            return [graph.Path.from_neo4j(path) for path in result.value()]

        return None

    # EXPAND NODE

    def expand_node(
        self, node: graph.Node, rel_types: t.Collection[str]
    ) -> t.Optional[t.List[graph.Path]]:
        with self._driver.session() as session:
            return session.read_transaction(
                self._expand_node, node, rel_types, self.lang
            )

    @staticmethod
    def _expand_node(
        tx: neo4j.Session, node: graph.Node, rel_types: t.Collection[str], lang: str
    ) -> t.Optional[t.List[graph.Path]]:
        rel_query = "|:".join(rel_types)
        result = tx.run(
            f"MATCH p=(n:Concept {{language: $lang}})-[r:{rel_query}]-(m:Concept {{language: $lang}})"
            f"WHERE id(n)={node.id} RETURN p",
            lang=lang,
        )

        if result:
            return [graph.Path.from_neo4j(path) for path in result.value()]

        return None
