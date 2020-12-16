import json
import logging
import os
from recap_argument_graph_adaptation.controller import evaluate
import typing as t
from pathlib import Path
from sklearn.model_selection import ParameterGrid

import pendulum

from .controller import adapt, export, extract, load
from .model import adaptation
from .model.config import config

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
log = logging.getLogger(__name__)


def _timestamp() -> str:
    return pendulum.now().format("YYYY-MM-DD-HH-mm-ss")


def run():
    log.info("Initializing.")
    cases = load.cases()
    out_path = Path(config["path"]["output"], _timestamp())

    param_grid = ParameterGrid(dict(config["tuning"]))

    for case in cases:
        for params in param_grid:
            config["_tuning"] = params

            adapted_concepts = {}

            if config["nlp"]["knowledge_graph"] == "wordnet":
                # for min_concept_score in range(20, 60, 10):
                #     config["nlp"]["min_concept_score"] = min_concept_score / 100
                adapted_concepts = _perform_wordnet_adaptation(case, out_path)

            elif config["nlp"]["knowledge_graph"] == "conceptnet":
                adaptation_methods = [adaptation.Method(config["conceptnet"]["method"])]
                adaptation_selectors = [
                    adaptation.Selector(config["conceptnet"]["selector"])
                ]

                if config["conceptnet"]["gridsearch"]:
                    adaptation_methods = [method for method in adaptation.Method]
                    adaptation_selectors = [
                        selector for selector in adaptation.Selector
                    ]

                for adaptation_method in adaptation_methods:
                    for adaptation_selector in adaptation_selectors:
                        adapted_concepts = _perform_conceptnet_adaptation(
                            case, adaptation_method, adaptation_selector, out_path
                        )


def _perform_wordnet_adaptation(
    case: adaptation.Case,
    out_path: Path,
) -> t.Dict[adaptation.Concept, adaptation.Concept]:
    log.info(f"Processing '{case.name}'.")

    nested_out_path: Path = out_path / case.name
    nested_out_path.mkdir(parents=True, exist_ok=True)

    adaptation_results = {}
    global_adapted_concepts = {}

    for rule in case.rules:
        log.info(f"Processing rule {str(rule)}.")

        concepts = extract.keywords(case.graph, rule)
        adapted_concepts, adapted_synsets = adapt.synsets(concepts, rule)
        adapt.argument_graph(case.graph, rule, adapted_concepts)

        adaptation_results[str(rule)] = export.statistic(
            concepts, {}, {}, adapted_synsets, adapted_concepts
        )
        global_adapted_concepts.update(adapted_concepts)

    eval_results = evaluate.case(case, global_adapted_concepts)

    stats = {
        "evaluation": eval_results.to_dict(),
        "results": adaptation_results,
        "config": dict(config),
    }
    _write_output(case, stats, nested_out_path)

    return global_adapted_concepts


def _perform_conceptnet_adaptation(
    case: adaptation.Case,
    adaptation_method: adaptation.Method,
    adaptation_selector: adaptation.Selector,
    out_path: Path,
) -> t.Dict[adaptation.Concept, adaptation.Concept]:
    log.info(
        f"Processing '{case.name}' "
        f"with method '{adaptation_method.value}' "
        f"and selector '{adaptation_selector.value}'."
    )

    nested_out_path: Path = out_path / case.name

    if config["conceptnet"]["gridsearch"]:
        nested_out_path = (
            nested_out_path / adaptation_method.value / adaptation_selector.value
        )

    nested_out_path.mkdir(parents=True, exist_ok=True)

    adaptation_results = {}
    global_adapted_concepts = {}

    for rule in case.rules:
        log.info(f"Processing rule {str(rule)}.")

        concepts = extract.keywords(case.graph, rule)
        reference_paths = extract.paths(concepts, rule, adaptation_method)
        adapted_concepts, adapted_paths = adapt.paths(
            reference_paths, rule, adaptation_selector, adaptation_method
        )
        adapt.argument_graph(case.graph, rule, adapted_concepts)

        adaptation_results[str(rule)] = export.statistic(
            concepts, reference_paths, adapted_paths, {}, adapted_concepts
        )
        global_adapted_concepts.update(adapted_concepts)

    stats = {"results": adaptation_results, "config": dict(config)}
    _write_output(case, stats, nested_out_path)

    return global_adapted_concepts


def _write_output(
    case: adaptation.Case, stats: t.Mapping[str, t.Any], path: Path
) -> None:
    case.graph.save(path / "case.json")
    case.graph.render(path / "case.pdf")
    stats_path = path / "stats.json"

    with stats_path.open("w") as file:
        json.dump(
            stats,
            file,
            ensure_ascii=False,
            indent=4,
            # default=lambda x: str(x),
        )


if __name__ == "__main__":
    run()
