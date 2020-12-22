import enum
import itertools
import requests
import stackprinter
import statistics
from collections import defaultdict
import uvicorn
import json
import logging
import os
import multiprocessing
from recap_argument_graph_adaptation.model.evaluation import Evaluation
from recap_argument_graph_adaptation.controller import evaluate
import typing as t
from pathlib import Path
from sklearn.model_selection import ParameterGrid
import time

import pendulum

from .controller import adapt, export, extract, load
from .model import adaptation
from .model.config import config

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
log = logging.getLogger(__name__)
stackprinter.set_excepthook(style="darkbg2")


def _timestamp() -> str:
    return pendulum.now().format("YYYY-MM-DD-HH-mm-ss")


def _file_path(path: Path) -> str:
    return "file://" + str(path)


# https://stackoverflow.com/questions/53321925/use-nltk-corpus-multithreaded


def run():
    log.info("Initializing.")
    wn_server = multiprocessing.Process(
        target=uvicorn.run,
        args=("recap_argument_graph_adaptation.wn_server:app",),
        kwargs={
            "host": config["worndet"]["host"],
            "port": config["wordnet"]["port"],
            "log_level": "warning",
        },
        daemon=True,
    )
    wn_server.start()
    wn_server_ready = False

    while not wn_server_ready:
        try:
            response = requests.get(
                f"http://{config['worndet']['host']}:{config['wordnet']['port']}"
            )

            if response.ok:
                wn_server_ready = True

        except requests.ConnectionError:
            time.sleep(0.5)

    out_path = Path(config["path"]["output"], _timestamp())
    cases = load.cases()

    param_grid = list(ParameterGrid(dict(config["tuning"])))

    run_args = (
        (i, params, len(param_grid), case, out_path)
        for (i, params), case in itertools.product(enumerate(param_grid), cases)
    )

    processes = None if config["processes"] == 0 else int(config["processes"])

    if processes == 1:
        raw_results = [_multiprocessing_run(*run_arg) for run_arg in run_args]
    else:
        with multiprocessing.Pool(processes) as pool:
            raw_results = pool.starmap(_multiprocessing_run, run_args)

    case_results = defaultdict(list)
    param_results = [[] for _ in range(len(param_grid))]

    for case, i, score in raw_results:
        case_results[case].append((score, i))
        param_results[i].append(score)

    case_max_results = {
        case: max(scores, key=lambda x: x[0]) for case, scores in case_results.items()
    }
    best_case_results = {}

    for case, (score, i) in case_max_results.items():
        current_path = _nested_path(
            out_path / case, len(param_grid), param_grid[i]
        ).resolve()
        best_case_results[case] = {
            "max_score": score,
            "case.json": _file_path(current_path / "case.json"),
            "case.pdf": _file_path(current_path / "case.pdf"),
            "stats.json": _file_path(current_path / "stats.json"),
            "config": param_grid[i],
        }

    mean_param_results = sorted(
        [
            {"mean_score": statistics.mean(scores), "config": param_grid[i]}
            for i, scores in enumerate(param_results)
        ],
        key=lambda x: x["mean_score"],
        reverse=True,
    )

    grid_stats_path = out_path / "grid_stats.json"
    grid_stats = {
        "param_results": mean_param_results,
        "case_results": best_case_results,
    }

    with grid_stats_path.open("w") as file:
        _json_dump(grid_stats, file)


def _multiprocessing_run(
    i: int,
    params: t.Mapping[str, t.Any],
    total_params: int,
    case: adaptation.PlainCase,
    out_path: Path,
) -> t.Tuple[str, int, float]:
    config["_tuning"] = params
    config["_tuning_runs"] = total_params
    case_nlp = case.nlp(load.spacy_nlp())
    eval_result = _perform_adaptation(case_nlp, out_path)
    return (str(case_nlp), i, eval_result.score)


def _nested_path(
    path: Path, total_runs: int, nested_folders: t.Mapping[str, t.Any]
) -> Path:
    nested_path = path

    if total_runs > 1:
        for tuning_key, tuning_value in nested_folders.items():
            nested_path /= f"{tuning_key}_{tuning_value}"

    return nested_path


def _perform_adaptation(
    case: adaptation.Case,
    out_path: Path,
) -> Evaluation:
    log.info(
        f"Processing '{case.name}' with rules {[str(rule) for rule in case.rules]}."
    )

    nested_out_path = _nested_path(
        out_path / case.name, config["_tuning_runs"], config["_tuning"]
    )
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


# TODO: On the GPU server, multiprocessing fails at the end.


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
