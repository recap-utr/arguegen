import json
import logging
from pathlib import Path
import typing as t

import recap_argument_graph as ag
import spacy

from .controller import adapt, extract, load, export
from .model import graph, adaptation
from .model.config import config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run():
    log.info("Initializing.")
    cases = load.cases()

    for case in cases:
        concepts = extract.keywords(case.graph)
        # log.info(
        #     f"Found the following concepts: {', '.join((str(concept) for concept in concepts))}"
        # )

        adaptation_methods = [adaptation.Method(config["adaptation"]["method"])]
        adaptation_selectors = [adaptation.Selector(config["adaptation"]["selector"])]

        if config["adaptation"]["gridsearch"]:
            adaptation_methods = [method for method in adaptation.Method]
            adaptation_selectors = [selector for selector in adaptation.Selector]

        for adaptation_method in adaptation_methods:
            for adaptation_selector in adaptation_selectors:
                _perform_adaptation(
                    case, concepts, adaptation_method, adaptation_selector
                )


def _perform_adaptation(
    case: adaptation.Case,
    concepts: t.Set[adaptation.Concept],
    adaptation_method: adaptation.Method,
    adaptation_selector: adaptation.Selector,
) -> None:
    log.info(
        f"Processing '{case.graph.name}' "
        f"with method '{adaptation_method.value}' "
        f"and selector '{adaptation_selector.value}'."
    )

    out_path = Path(
        config["path"]["output"], adaptation_method.value, adaptation_selector.value,
    )

    adaptation_results = {}

    for rule in case.rules:
        log.info(f"Processing rule ({rule[0]})->({rule[1]}).")

        reference_paths = extract.paths(concepts, rule, adaptation_method)
        adapted_concepts, adapted_paths = adapt.paths(
            reference_paths, rule, adaptation_selector, adaptation_method
        )
        adapt.argument_graph(case.graph, rule, adapted_concepts)

        adaptation_results[f"({rule[0]})->({rule[1]})"] = export.statistic(
            concepts, reference_paths, adapted_concepts, adapted_paths
        )

    case.graph.render(out_path)
    case.graph.save(out_path)
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
