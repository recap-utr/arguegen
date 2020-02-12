import json
import logging
from pathlib import Path

import recap_argument_graph as ag
import spacy

from .controller import adapt, concept, load
from .model import adaptation
from .model.config import config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__package__)

nlp = spacy.load(config["spacy"]["model"])

cases = load.cases()
out_path = Path(
    config["path"]["output"],
    config["adaptation"]["method"],
    config["adaptation"]["selector"],
)

for case in cases:
    concepts = concept.from_graph(case.graph)
    adaptation_results = {}
    stats_path = out_path / f"{case.graph.name}.json"

    for rule in case.rules:
        reference_paths = concept.paths(concepts, rule)
        adapted_concepts = adapt.paths(reference_paths, rule)

        adapt.argument_graph(case.graph, rule, adapted_concepts)
        adaptation_results[f"{rule[0]}->{rule[1]}"] = adapted_concepts

    case.graph.render(out_path)

    with stats_path.open("w") as f:
        json.dump(adaptation_results, f, ensure_ascii=False, indent=4)
