import neo4j
from .config import config
from . import conceptnet

class Database:
    _driver: neo4j.Driver
    lang: str

    def __init__(self, lang: str):
        self._driver = neo4j.GraphDatabase.driver(config["neo4j"]["url"], auth=None, encrypted=False)  # auth=("username", "password")
        self.lang = lang

    def get_concept(self, name: str):
        with self._driver.session() as session:
            return session.read_transaction(self.match_concept_node, name, self.lang)

    @staticmethod
    def match_concept_node(tx: neo4j.Session, name: str, lang: str):
        concept_name = conceptnet.concept_name(name, lang)
        result = tx.run("MATCH (n:Concept {name: $name, language: $lang}) RETURN n", name=concept_name, lang=lang).single()
        node = conceptnet.Node.from_neo4j(result["n"])

        return node
