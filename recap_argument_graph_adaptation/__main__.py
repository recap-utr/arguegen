from pathlib import Path

import recap_argument_graph as ag
import spacy
import logging

from .controller import concept, adapt
from .model.config import config
from .model import adaptation

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__package__)


nlp = spacy.load("en_core_web_lg")

case_graph = ag.Graph.open(Path("data/case-base/nodeset6366.json"), nlp)
adapted_graph = ag.Graph.open(Path("data/case-base/nodeset6366.json"))

concepts = concept.from_graph(case_graph)

adaptation_rules = [("death penalty", "punishment")]
method = adaptation.Method(config["adaptation_method"])

for rule in adaptation_rules:
    reference_paths = concept.paths(concepts, rule, method)
    adapted_paths = adapt.paths(reference_paths, rule, method)
    adapt.argument_graph(adapted_graph, adapted_paths)

adapted_graph.render(Path("data/out/"))
