import neo4j
from .config import config
from . import conceptnet
from gibberish import Gibberish
import typing as t


class Database:
    _driver: neo4j.Driver
    lang: str

    def __init__(self, lang: str):
        self._driver = neo4j.GraphDatabase.driver(
            config["neo4j"]["url"], auth=None, encrypted=False
        )  # auth=("username", "password")
        self.lang = lang

    # NODE

    def node(self, name: str) -> t.Optional[neo4j.Node]:
        with self._driver.session() as session:
            return session.read_transaction(self._node, name, self.lang)

    @staticmethod
    def _node(tx: neo4j.Session, name: str, lang: str) -> t.Optional[neo4j.Node]:
        result = tx.run(
            "MATCH (n:Concept {name: $name, language: $lang}) RETURN n",
            name=conceptnet.concept_name(name, lang),
            lang=lang,
        ).single()

        return result.value() if result else None

    # SHORTEST PATH

    def shortest_path(self, start: str, end: str) -> t.Optional[neo4j.Path]:
        with self._driver.session() as session:
            return session.read_transaction(self._shortest_path, start, end, self.lang)

    @staticmethod
    def _shortest_path(
        tx: neo4j.Session, start: str, end: str, lang: str
    ) -> t.Optional[neo4j.Path]:
        result = tx.run(
            "MATCH p = shortestPath((n:Concept {name: $start, language: $lang})-[*..10]-"
            "(m:Concept {name: $end, language: $lang})) RETURN p",
            start=conceptnet.concept_name(start, lang),
            end=conceptnet.concept_name(end, lang),
            # max_relations=max_relations,
            lang=lang,
        ).single()

        return result.value() if result else None

    # SINGLE PATH

    def single_path(self, name: str) -> t.Optional[neo4j.Path]:
        with self._driver.session() as session:
            return session.read_transaction(self._single_path, name, self.lang)

    @staticmethod
    def _single_path(tx: neo4j.Session, name: str, lang: str):
        result = tx.run(
            "MATCH p=(n:Concept {name: $name, language: $lang}) RETURN p",
            name=conceptnet.concept_name(name, lang),
            lang=lang,
        ).single()

        return result.value() if result else None

    # EXTEND PATH

    def extend_path(
        self, current_path: neo4j.Path, extension_path: neo4j.Path
    ) -> neo4j.Path:
        with self._driver.session() as session:
            return session.read_transaction(self._extend_path, current_path, extension_path)

    @staticmethod
    def _extend_path(
        tx: neo4j.Session, current_path: neo4j.Path, extension_path: neo4j.Path
    ) -> neo4j.Path:
        wordgen = Gibberish()
        relationships = list(current_path.relationships) + list(extension_path.relationships)
        words = wordgen.generate_words(len(relationships))

        q = "MATCH p = ( (:Concept)-["
        q += "]-(:Concept)-[".join(words)
        q += "]-(:Concept)) WHERE "

        where_expr = [f"id({word})={rel.id}" for word, rel in zip(words, relationships)]
        q += " AND ".join(where_expr)

        q += " RETURN p"

        result = tx.run(q).single()

        return result.value()

    # EXPAND NODE

    def expand_node(
        self, node: neo4j.Node, rel_types: t.Collection[str]
    ) -> t.Optional[t.List[neo4j.Path]]:
        with self._driver.session() as session:
            return session.read_transaction(
                self._expand_node, node, rel_types, self.lang
            )

    @staticmethod
    def _expand_node(
        tx: neo4j.Session, node: neo4j.Node, rel_types: t.Collection[str], lang: str
    ) -> t.Optional[t.List[neo4j.Path]]:
        rel_query = "|:".join(rel_types)
        result = tx.run(
            f"MATCH p=(n:Concept {{language: $lang}})-[r:{rel_query}]-(m:Concept {{language: $lang}})"
            f"WHERE id(n)={node.id} RETURN p",
            lang=lang,
        )

        return result.value() if result else None
