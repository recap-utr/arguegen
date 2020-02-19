import json
import logging
from pathlib import Path
import typing as t

import recap_argument_graph as ag
import spacy

from .controller import adapt, extract, load, export
from .model.adaptation import Concept
from .model import graph
from .model.config import config

logging.basicConfig(level=logging.INFO)
logging.getLogger(__package__).setLevel(logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

nlp = spacy.load(config["spacy"]["model"])

# TODO: Many adapted paths have shorter lengths and thus are not considered!


def run():
    cases = load.cases()
    out_path = Path(
        config["path"]["output"],
        config["adaptation"]["method"],
        config["adaptation"]["selector"],
    )

    for case in cases:
        log.info(f"Processing '{case.graph.name}'.")

        concepts = extract.keywords(case.graph)
        log.info(
            f"Found the following concepts: {', '.join((str(concept) for concept in concepts))}"
        )

        adaptation_results = {}

        for rule in case.rules:
            reference_paths = extract.paths(concepts, rule)
            adapted_concepts, adapted_paths = adapt.paths(reference_paths, rule)

            adapt.argument_graph(case.graph, rule, adapted_concepts)
            adaptation_results[f"{rule[0]}->{rule[1]}"] = export.statistic(
                concepts, reference_paths, adapted_concepts, adapted_paths
            )

        case.graph.save(out_path)
        case.graph.render(out_path)
        stats_path = out_path / f"{case.graph.name}-stats.json"

        with stats_path.open("w") as file:
            json.dump(
                adaptation_results,
                file,
                ensure_ascii=False,
                indent=4,
                default=lambda x: str(x),
                # default=lambda x: x.__dict__,
            )


if __name__ == "__main__":
    run()
