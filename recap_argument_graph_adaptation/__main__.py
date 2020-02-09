import json
from pathlib import Path

import recap_argument_graph as ag
import spacy
import logging

from .controller import concept, adapt
from .model.config import config
from .model import adaptation

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__package__)

nlp = spacy.load(config["spacy"]["model"])

case_graph = ag.Graph.open(Path("data/case-base/nodeset6366.json"), nlp)
adapted_graph = ag.Graph.open(Path("data/case-base/nodeset6366.json"))
out_path = Path(
    "data", "out", config["adaptation"]["method"], config["adaptation"]["selector"],
)

concepts = concept.from_graph(case_graph)

adaptation_rules = {"death penalty": "punishment"}
adaptation_results = {}

for rule in adaptation_rules.items():
    reference_paths = concept.paths(concepts, rule)
    adapted_concepts = adapt.paths(reference_paths, rule)

    adapt.argument_graph(adapted_graph, rule, adapted_concepts)
    adaptation_results[f"{rule[0]}->{rule[1]}"] = adapted_concepts

adapted_graph.render(out_path)
stats_path = out_path / f"{adapted_graph.name}.json"

with stats_path.open("w") as f:
    json.dump(adaptation_results, f, ensure_ascii=False, indent=4)
