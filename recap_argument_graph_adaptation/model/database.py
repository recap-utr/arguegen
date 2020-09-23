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
            auth=(config["neo4j"]["username"], config["neo4j"]["password"]),
            encrypted=False,
        )

        if isinstance(driver, neo4j.Neo4jDriver):
            self._driver = driver
            self.lang = config["nlp"]["lang"]

    # NODE

    def node(self, name: str, pos: graph.POS) -> t.Optional[graph.Node]:
        with self._driver.session() as session:
            return session.read_transaction(self._node, name, pos, self.lang)

    @staticmethod
    def _node(
        tx: neo4j.Session, name: str, pos: graph.POS, lang: str
    ) -> t.Optional[graph.Node]:
        query = "MATCH (n:Concept {name: $name, pos: $pos, language: $lang}) RETURN n"

        # We follow all available 'FormOf' relations to simplify path construction
        if config["neo4j"]["concept_root_form"]:
            query = "MATCH p=((n:Concept {name: $name, pos: $pos, language: $lang})-[:FormOf*0..]->(m:Concept {language: $lang})) RETURN m ORDER BY length(p) DESC LIMIT 1"

        # First, run the query with the given POS.
        # If no concept is found, retry the query without specifying a POS.
        record = tx.run(
            query,
            name=conceptnet.concept_name(name, lang),
            pos=pos.value,
            lang=lang,
        ).single()

        if not record:
            record = tx.run(
                query,
                name=conceptnet.concept_name(name, lang),
                pos=graph.POS.OTHER.value,
                lang=lang,
            ).single()

        return graph.Node.from_neo4j(record.value()) if record else None

    # SHORTEST PATH

    def shortest_path(
        self, start: graph.Node, end: graph.Node
    ) -> t.Optional[graph.Path]:
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
        start: graph.Node,
        end: graph.Node,
        relation_types: t.Collection[str],
        max_relations: int,
        lang: str,
    ) -> t.Optional[graph.Path]:
        rel_query = _aggregate_relations(relation_types)

        query = (
            "MATCH p = shortestPath((n:Concept {name: $start_name, pos: $start_pos, language: $lang})"
            f"-[{rel_query}*..{max_relations}]{_arrow()}"
            "(m:Concept {name: $end_name, pos: $end_pos, language: $lang})) RETURN p"
        )

        record = tx.run(
            query,
            start_name=start.name,
            start_pos=start.pos.value,
            end_name=end.name,
            end_pos=end.pos.value,
            lang=lang,
        ).single()

        if not record:
            record = tx.run(
                query,
                start_name=start.name,
                start_pos=graph.POS.OTHER.value,
                end_name=end.name,
                end_pos=graph.POS.OTHER.value,
                lang=lang,
            ).single()

        return graph.Path.from_neo4j(record.value()) if record else None

    # ALL SHORTEST PATHS

    def all_shortest_paths(
        self, start: graph.Node, end: graph.Node
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
        start: graph.Node,
        end: graph.Node,
        relation_types: t.Collection[str],
        max_relations: int,
        lang: str,
    ) -> t.Optional[t.List[graph.Path]]:
        rel_query = _aggregate_relations(relation_types)

        query = (
            "MATCH p = allShortestPaths((n:Concept {name: $start_name, pos: $start_pos, language: $lang})"
            f"-[{rel_query}*..{max_relations}]{_arrow()}"
            "(m:Concept {name: $end_name, pos: $end_pos, language: $lang})) RETURN p"
        )

        records = tx.run(
            query,
            start_name=start.name,
            start_pos=start.pos.value,
            end_name=end.name,
            end_pos=end.pos.value,
            lang=lang,
        )

        if not records:
            records = tx.run(
                query,
                start_name=start.name,
                start_pos=graph.POS.OTHER.value,
                end_name=end.name,
                end_pos=graph.POS.OTHER.value,
                lang=lang,
            )

        if records:
            return [graph.Path.from_neo4j(record.value()) for record in records]

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
            f"MATCH p=(n:Concept)-[r{rel_query}]{_arrow()}(m:Concept {{language: $lang}})"
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


def _arrow() -> str:
    return "->" if config["neo4j"]["directed_relations"] else "-"
