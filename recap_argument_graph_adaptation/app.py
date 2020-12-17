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
            _perform_adaptation(case, out_path)


def _perform_adaptation(
    case: adaptation.Case,
    out_path: Path,
) -> t.Dict[adaptation.Concept, adaptation.Concept]:
    log.info(f"Processing '{case.name}'.")

    nested_out_path: Path = out_path / case.name
    # TODO: Due to the parameter grid, a sensible folder structure has to be created
    # nested_out_path = (
    #         nested_out_path / adaptation_method.value / adaptation_selector.value
    #     )
    nested_out_path.mkdir(parents=True, exist_ok=True)

    adaptation_results = {}
    global_adapted_concepts = {}

    for rule in case.rules:
        log.info(f"Processing rule {str(rule)}.")

        adapted_concepts = {}
        reference_paths = {}
        adapted_paths = {}
        adapted_synsets = {}

        concepts = extract.keywords(case.graph, rule)

        if config["nlp"]["knowledge_graph"] == "wordnet":
            adapted_concepts, adapted_synsets = adapt.synsets(concepts, rule)
        elif config["nlp"]["knowledge_graph"] == "conceptnet":
            reference_paths = extract.paths(concepts, rule)
            adapted_concepts, adapted_paths = adapt.paths(reference_paths, rule)

        adapt.argument_graph(case.graph, rule, adapted_concepts)

        adaptation_results[str(rule)] = export.statistic(
            concepts, reference_paths, adapted_paths, adapted_synsets, adapted_concepts
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
