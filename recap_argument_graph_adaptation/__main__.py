import typing as t
from dataclasses import dataclass
from pathlib import Path

import neo4j
import recap_argument_graph as ag
import spacy
import stackprinter

from .controller import concept
from .model import conceptnet
from .model.adaptation import ConceptAdaptation
from .model.config import Config
from .model.database import Database

config = Config.instance()
db = Database("en")
nlp = spacy.load("en")

stackprinter.set_excepthook(style="darkbg")


# TODO: allShortestPaths could be used to filter for paths that contain names of existing concepts in graph
# TODO: Selection of the final adapted relationship is random: next()

case_graph = ag.Graph.open(Path("data/case-base/nodeset6366.json"), nlp)
adapted_graph = ag.Graph.open(Path("data/case-base/nodeset6366.json"))
query_graph = ag.Graph("Query")
query_graph.add_node(
    ag.Node(
        query_graph.keygen(),
        nlp(
            "Any punishment is a legal means that as such is not practicable in Germany."
        ),
        ag.NodeCategory.I,
    )
)

case_concepts = concept.from_graph(case_graph)
query_concepts = concept.from_graph(query_graph)

adaptation_rules = [("penalty", "punishment"), ("death", "any")]

for rule in adaptation_rules:
    original_paths = concept.reference_paths(db, case_concepts, rule)
    adapted_paths = concept.adapt_paths(db, original_paths, rule)
    concept.adapt_graph(adapted_graph, adapted_paths)

adapted_graph.render(Path("data/out/"))
