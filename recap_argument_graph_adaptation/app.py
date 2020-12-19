import statistics
from collections import defaultdict
import json
import logging
import os
from recap_argument_graph_adaptation.model.evaluation import Evaluation
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

    param_grid = list(ParameterGrid(dict(config["tuning"])))
    case_results = defaultdict(list)
    param_results = []

    for i, params in enumerate(param_grid):
        config["_tuning"] = params
        current_results = []

        for case in cases:
            eval_result = _perform_adaptation(case, out_path)
            case_results[case].append((eval_result, i))
            current_results.append(eval_result)

        param_results.append(current_results)

    case_scores = {
        case: max(results, key=lambda x: x[0]) for case, results in case_results.items()
    }
    best_case_results = {
        str(case): {"max_score": result.score, "config": param_grid[i]}
        for case, (result, i) in case_scores.items()
    }

    param_scores = [[result.score for result in results] for results in param_results]
    mean_param_results = sorted(
        [
            {"mean_score": statistics.mean(scores), "config": param_grid[i]}
            for i, scores in enumerate(param_scores)
        ],
        key=lambda x: x["mean_score"],
    )

    grid_stats_path = out_path / "grid_stats.json"
    grid_stats = {
        "param_results": mean_param_results,
        "case_results": best_case_results,
    }

    with grid_stats_path.open("w") as file:
        _json_dump(grid_stats, file)


def _perform_adaptation(
    case: adaptation.Case,
    out_path: Path,
) -> Evaluation:
    log.info(
        f"Processing '{case.name}' with rules {[str(rule) for rule in case.rules]}."
    )

    nested_out_path: Path = out_path / case.name
    # TODO: Due to the parameter grid, a sensible folder structure has to be created
    # nested_out_path = (
    #         nested_out_path / adaptation_method.value / adaptation_selector.value
    #     )
    nested_out_path.mkdir(parents=True, exist_ok=True)

    adapted_concepts = {}
    reference_paths = {}
    adapted_paths = {}
    adapted_synsets = {}

    concepts = extract.keywords(case.graph, case.rules)

    if config["nlp"]["knowledge_graph"] == "wordnet":
        adapted_concepts, adapted_synsets = adapt.synsets(concepts, case.rules)
    elif config["nlp"]["knowledge_graph"] == "conceptnet":
        reference_paths = extract.paths(concepts, case.rules)
        adapted_concepts, adapted_paths = adapt.paths(reference_paths, case.rules)

    adapt.argument_graph(case.graph, case.rules, adapted_concepts)

    adaptation_results = export.statistic(
        concepts, reference_paths, adapted_paths, adapted_synsets, adapted_concepts
    )
    eval_results = evaluate.case(case, adapted_concepts)

    stats_export = {
        "evaluation": eval_results.to_dict(),
        "results": adaptation_results,
        "config": dict(config),
    }
    _write_output(case, stats_export, nested_out_path)

    return eval_results


def _write_output(
    case: adaptation.Case, stats: t.Mapping[str, t.Any], path: Path
) -> None:
    case.graph.save(path / "case.json")
    case.graph.render(path / "case.pdf")
    stats_path = path / "stats.json"

    with stats_path.open("w") as file:
        _json_dump(stats, file)


def _json_dump(mapping: t.Mapping[str, t.Any], file: t.TextIO) -> None:
    json.dump(
        mapping,
        file,
        ensure_ascii=False,
        indent=4,
    )


if __name__ == "__main__":
    run()
