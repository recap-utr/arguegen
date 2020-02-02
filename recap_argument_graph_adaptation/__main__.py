from pathlib import Path

import recap_argument_graph as ag
import spacy
import logging

from .controller import concept, adapt
from .model.config import Config
from .model.database import Database

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__package__)


config = Config.instance()
nlp = spacy.load("en_core_web_lg")

case_graph = ag.Graph.open(Path("data/case-base/nodeset6366.json"), nlp)
adapted_graph = ag.Graph.open(Path("data/case-base/nodeset6366.json"))

concepts = concept.from_graph(case_graph)

adaptation_rules = [("death penalty", "punishment")]

for rule in adaptation_rules:
    reference_paths = concept.paths(concepts, rule, "between")
    adapted_paths = adapt.paths(reference_paths, rule, "between")
    adapt.argument_graph(adapted_graph, adapted_paths)

adapted_graph.render(Path("data/out/"))
