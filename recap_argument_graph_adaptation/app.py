import itertools
import json
import logging
import multiprocessing
import shutil
import statistics
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

import pendulum
import recap_argument_graph as ag
import requests
import typer
from sklearn.model_selection import ParameterGrid

from recap_argument_graph_adaptation.controller import evaluate, spacy, wordnet
from recap_argument_graph_adaptation.helper import convert
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


# TODO: Refactor files/code structure

# https://stackoverflow.com/a/50379950/7626878
def init_child(lock_):
    wordnet.lock = lock_


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
    start_time = timer()

    out_path = Path(config["resources"]["cases"]["output"], _timestamp())
    cases = load.cases()

    param_grid = [
        params
        for params in ParameterGrid(dict(config["tuning"]))
        if round(sum(_filter_mapping(params, "weight").values()), 2) == 1
        and round(sum(_filter_mapping(params, "score").values()), 2) == 1
    ]
    if not param_grid:
        param_grid = [{key: values[0] for key, values in config["tuning"].items()}]

    total_params = len(param_grid)
    total_cases = len(cases)
    total_runs = total_params * total_cases

    run_args = [
        RunArgs(
            i,
            total_runs,
            i_params,
            total_params,
            params,
            i_case,
            total_cases,
            case,
            out_path,
        )
        for i, ((i_params, params), (i_case, case)) in enumerate(
            itertools.product(enumerate(param_grid), enumerate(cases))
        )
    ]

    processes = (
        multiprocessing.cpu_count() - 1
        if config["resources"]["processes"] == 0
        else int(config["resources"]["processes"])
    )
    results = []

    log.info(f"Starting with {len(run_args)} runs using {processes} processes.")

    if processes == 1 or len(run_args) == 1:
        logging.getLogger(__package__).setLevel(logging.DEBUG)
        results = [_multiprocessing_run(run_arg) for run_arg in run_args]
    else:
        local_lock = multiprocessing.Lock()
        with multiprocessing.Pool(
            processes, initializer=init_child, initargs=(local_lock,)
        ) as pool:
            with typer.progressbar(
                pool.imap(_multiprocessing_run, run_args),
                length=len(run_args),
                show_percent=True,
                show_pos=True,
                show_eta=True,
            ) as iterator:
                for result in iterator:
                    results.append(result)

    end_time = timer()
    duration = end_time - start_time

    if config["adaptation"]["export_grid_stats"] and len(run_args) > 1:
        _grid_stats(results, duration, param_grid, out_path)

    log.info(f"Finished in {duration} sec.")


def _output_file_paths(parent_folder: Path) -> t.Dict[str, str]:
    paths = {}
    filenames = []

    if config["adaptation"]["export_graph"]:
        filenames.extend(("case.json", "case.pdf"))

    if config["adaptation"]["export_single_stats"]:
        filenames.append("stats.json")

        for filename in filenames:
            paths[filename] = _file_path(parent_folder / filename)

    return paths


def _grid_stats(
    results: t.Iterable[t.Tuple[str, int, Evaluation]],
    duration: float,
    param_grid: t.Sequence[t.Mapping[str, t.Any]],
    out_path: Path,
) -> None:
    log.info("Exporting grid stats.")

    results = [entry for entry in results if entry is not None]
    case_results = defaultdict(list)
    param_results = [[] for _ in range(len(param_grid))]

    for case, i, eval in results:
        case_results[case].append((eval, i))
        param_results[i].append(eval.score)

    best_case_results = {}

    for case, eval_results in case_results.items():
        eval_results.sort(key=lambda x: x[0].score, reverse=True)
        _results = []

        for eval, i in eval_results:
            current_path = _nested_path(
                out_path / case, len(param_grid), param_grid[i]
            ).resolve()

            # Move the best results to the root folder for that case.
            if len(_results) == 0:
                for file in ("case.json", "case.pdf", "stats.json"):
                    try:
                        shutil.copy(
                            current_path / file, out_path / case / f"best_{file}"
                        )
                    except Exception:
                        pass

            _results.append(
                {
                    "evaluation": eval.to_dict(compact=True),
                    "files": _output_file_paths(current_path),
                    "config": param_grid[i],
                }
            )

        best_case_results[case] = _results

    mean_param_results = []

    for i, scores in enumerate(param_results):
        if scores:
            current_cases = {}

            for case, eval_results in case_results.items():
                eval_result = next(filter(lambda x: x[1] == i, eval_results), None)
                current_path = _nested_path(
                    out_path / case, len(param_grid), param_grid[i]
                ).resolve()
                case_eval_output = None

                if eval_result:
                    case_eval_output = eval_result[0].to_dict(compact=True)

                current_cases[case] = {
                    "evaluation": case_eval_output,
                    "files": _output_file_paths(current_path),
                }

            mean_param_results.append(
                {
                    "mean_score": statistics.mean(scores),
                    "config": param_grid[i],
                    "cases": current_cases,
                }
            )

    mean_param_results.sort(
        key=lambda x: x["mean_score"],
        reverse=True,
    )

    grid_stats_path = out_path / "grid_stats.json"
    grid_stats = {
        "duration": duration,
        "param_results": mean_param_results,
        "case_results": best_case_results,
    }

    grid_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with grid_stats_path.open("w") as file:
        _json_dump(grid_stats, file)


@dataclass()
class RunArgs:
    current_run: int
    total_runs: int
    current_params: int
    total_params: int
    params: t.Mapping[str, t.Any]
    current_case: int
    total_cases: int
    case: adaptation.Case
    out_path: Path


def _multiprocessing_run(args: RunArgs) -> t.Tuple[str, int, Evaluation]:
    log.debug(f"Starting run {args.current_run + 1}/{args.total_runs}.")

    config["_tuning"] = args.params
    config["_tuning_runs"] = args.total_runs
    spacy.session = requests.Session()

    log.debug("Starting adaptation pipeline.")
    eval_result = _perform_adaptation(args.case, args.out_path)

    return (str(args.case), args.current_params, eval_result)


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
    start_time = timer()
    nested_out_path = _nested_path(
        out_path / case.name, config["_tuning_runs"], config["_tuning"]
    )

    adapted_concepts = {}
    reference_paths = {}
    adapted_paths = {}
    adapted_synsets = {}
    adapted_graph = None

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
        adapted_graph = adapt.argument_graph(case.graph, case.rules, adapted_concepts)

    log.debug("Evaluating adaptations.")
    eval_results = evaluate.case(case, adapted_concepts)

    end_time = timer()

    log.debug("Exporting statistics.")
    adaptation_results = export.statistic(
        concepts, reference_paths, adapted_paths, adapted_synsets, adapted_concepts
    )

    rules_export = {
        "benchmark_rules": convert.list_str(case.benchmark_rules),
        "case_rules": convert.list_str(case.rules),
    }

    stats_export = {
        "evaluation": eval_results.to_dict(),
        "time": end_time - start_time,
        "rules": rules_export,
        "results": adaptation_results,
        "config": dict(config),
    }
    _write_output(adapted_graph, stats_export, nested_out_path)

    return eval_results


def _write_output(
    adapted_graph: t.Optional[ag.Graph], stats: t.Mapping[str, t.Any], path: Path
) -> None:
    path.mkdir(parents=True, exist_ok=True)

    if config["adaptation"]["export_graph"] and adapted_graph:
        adapted_graph.save(path / "case.json")
        adapted_graph.render(path / "case.pdf")

    if config["adaptation"]["export_single_stats"]:
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
