import neo4j
from .config import config
from . import conceptnet
import typing as t


class Database:
    _driver: neo4j.Driver
    lang: str

    def __init__(self, lang: str):
        self._driver = neo4j.GraphDatabase.driver(
            config["neo4j"]["url"], auth=None, encrypted=False
        )  # auth=("username", "password")
        self.lang = lang

    def get_concept(self, name: str) -> t.Optional[neo4j.Node]:
        with self._driver.session() as session:
            return session.read_transaction(self.match_concept_node, name, self.lang)

    def get_shortest_path(self, start: str, end: str) -> t.Optional[neo4j.Path]:
        with self._driver.session() as session:
            return session.read_transaction(
                self.match_shortest_path, start, end, self.lang
            )

    @staticmethod
    def match_concept_node(
        tx: neo4j.Session, name: str, lang: str
    ) -> t.Optional[neo4j.Node]:
        result = tx.run(
            "MATCH (n:Concept {name: $name, language: $lang}) RETURN n",
            name=conceptnet.concept_name(name, lang),
            lang=lang,
        ).single()

        return result.value() if result else None

    @staticmethod
    def match_shortest_path(
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
