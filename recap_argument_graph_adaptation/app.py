import itertools
import json
import logging
import multiprocessing
import statistics
import typing as t
from collections import defaultdict
from pathlib import Path

import pendulum
import requests
from sklearn.model_selection import ParameterGrid

from recap_argument_graph_adaptation.controller import evaluate, spacy, wordnet
from recap_argument_graph_adaptation.model.evaluation import Evaluation

from .controller import adapt, export, extract, load
from .model import adaptation
from .model.config import config

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
log = logging.getLogger(__name__)
# stackprinter.set_excepthook(style="darkbg2")


def _timestamp() -> str:
    return pendulum.now().format("YYYY-MM-DD-HH-mm-ss")


def _file_path(path: Path) -> str:
    return "file://" + str(path)


def init_child(lock_):
    global lock
    lock = lock_


def _filter_mapping(
    mapping: t.Mapping[str, t.Any], prefix: str
) -> t.Mapping[str, t.Any]:
    prefix = f"{prefix}_"

    return {
        key[len(prefix) :]: value
        for key, value in mapping.items()
        if key.startswith(prefix)
    }


# https://stackoverflow.com/questions/53321925/use-nltk-corpus-multithreaded


def run():
    log.info("Initializing.")

    out_path = Path(config["resources"]["cases"]["output"], _timestamp())
    cases = load.cases()

    param_grid = [
        params
        for params in ParameterGrid(dict(config["tuning"]))
        if round(sum(_filter_mapping(params, "weight").values()), 2) == 1
        and round(sum(_filter_mapping(params, "score").values()), 2) == 1
    ]
    # lock = multiprocessing.Lock()

    run_args = [
        (i, params, len(param_grid), case, out_path)
        for (i, params), case in itertools.product(enumerate(param_grid), cases)
    ]

    processes = (
        multiprocessing.cpu_count() - 1
        if config["processes"] == 0
        else int(config["processes"])
    )

    if processes == 1 or len(run_args) == 1:
        logging.getLogger(__package__).setLevel(logging.DEBUG)
        log.info("Single run.")
        raw_results = [_multiprocessing_run(*run_arg) for run_arg in run_args]
    else:
        log.info(f"Starting with {len(run_args)} runs using {processes} processes.")
        with multiprocessing.Pool(
            processes
        ) as pool:  # , initializer=init_child, initargs=(lock,)
            raw_results = pool.starmap(_multiprocessing_run, run_args)

    log.info("Exporting grid stats.")
    wordnet.lock = None

    raw_results = [entry for entry in raw_results if entry is not None]
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
            if scores
        ],
        key=lambda x: x["mean_score"],
        reverse=True,
    )

    grid_stats_path = out_path / "grid_stats.json"
    grid_stats = {
        "param_results": mean_param_results,
        "case_results": best_case_results,
    }

    grid_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with grid_stats_path.open("w") as file:
        _json_dump(grid_stats, file)

    log.info("Finished.")


def _multiprocessing_run(
    i: int,
    params: t.Mapping[str, t.Any],
    total_runs: int,
    case: adaptation.Case,
    out_path: Path,
) -> t.Tuple[str, int, float]:
    log.debug(f"Starting run {i + 1}/{total_runs}.")

    config["_tuning"] = params
    config["_tuning_runs"] = total_runs
    wordnet.session = requests.Session()
    spacy.session = requests.Session()
    # wordnet.lock = lock

    log.debug("Starting adaptation pipeline.")
    eval_result = _perform_adaptation(case, out_path)

    log.info(f"Finished with run {i + 1}/{total_runs}.")

    return (str(case), i, eval_result.score)


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
    nested_out_path = _nested_path(
        out_path / case.name, config["_tuning_runs"], config["_tuning"]
    )

    adapted_concepts = {}
    reference_paths = {}
    adapted_paths = {}
    adapted_synsets = {}

    log.debug("Extracting keywords.")
    concepts = extract.keywords(case.graph, case.rules)

    log.debug("Adapting concepts.")
    if config["adaptation"]["knowledge_graph"] == "wordnet":
        adapted_concepts, adapted_synsets = adapt.synsets(concepts, case.rules)
    elif config["adaptation"]["knowledge_graph"] == "conceptnet":
        reference_paths = extract.paths(concepts, case.rules)
        adapted_concepts, adapted_paths = adapt.paths(reference_paths, case.rules)

    if config["adaptation"]["export_graph"]:
        log.debug("Exporting graph.")
        adapt.argument_graph(case.graph, case.rules, adapted_concepts)

    log.debug("Evaluating adaptations.")
    eval_results = evaluate.case(case, adapted_concepts)

    log.debug("Exporting statistics.")
    adaptation_results = export.statistic(
        concepts, reference_paths, adapted_paths, adapted_synsets, adapted_concepts
    )
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
    path.mkdir(parents=True, exist_ok=True)

    if config["adaptation"]["export_graph"]:
        case.graph.save(path / "case.json")
        case.graph.render(path / "case.pdf")

    with (path / "stats.json").open("w") as file:
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
