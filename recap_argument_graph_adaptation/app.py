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

from .controller import adapt, convert, evaluate, export, extract, load
from .model import casebase as cb
from .model import spacy, wordnet
from .model.config import Config

config = Config.instance()

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
log = logging.getLogger(__name__)
# stackprinter.set_excepthook(style="darkbg2")


def _init_child_process(lock_):
    # https://stackoverflow.com/a/50379950/7626878
    wordnet.lock = lock_


processes = (
    multiprocessing.cpu_count() - 1
    if config["resources"]["processes"] == 0
    else int(config["resources"]["processes"])
)


def run():
    log.info("Initializing.")
    start_time = timer()

    out_path = Path(
        config["resources"]["cases"]["output"],
        pendulum.now().format("YYYY-MM-DD-HH-mm-ss"),
    )
    cases = load.cases()
    param_grid = load.parameter_grid()
    run_args = load.run_arguments(param_grid, cases)

    log.info(f"Starting with {len(run_args)} runs using {processes} processes.")
    results = []

    if processes == 1 or len(run_args) == 1:
        logging.getLogger(__package__).setLevel(logging.DEBUG)
        results = [_parametrized_run(run_arg) for run_arg in run_args]
    else:
        local_lock = multiprocessing.Lock()
        with multiprocessing.Pool(
            processes, initializer=_init_child_process, initargs=(local_lock,)
        ) as pool:
            with typer.progressbar(
                pool.imap(_parametrized_run, run_args),
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
        export.grid_stats(results, duration, param_grid, out_path)

    log.info(f"Finished in {duration} sec.")


def _parametrized_run(args: load.RunArgs) -> t.Tuple[str, int, cb.Evaluation]:
    log.debug(f"Starting run {args.current_run + 1}/{args.total_runs}.")

    config["_tuning"] = args.params
    config["_tuning_runs"] = args.total_runs
    spacy.session = requests.Session()
    case = args.case

    log.debug("Starting adaptation pipeline.")
    start_time = timer()
    nested_out_path = export.nested_path(
        args.out_path / case.name, config["_tuning_runs"], config["_tuning"]
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
    stats_export = {
        "evaluation": eval_results.to_dict(),
        "time": end_time - start_time,
        "rules": {
            "benchmark_rules": convert.list_str(case.benchmark_rules),
            "case_rules": convert.list_str(case.rules),
        },
        "results": export.statistic(
            concepts, reference_paths, adapted_paths, adapted_synsets, adapted_concepts
        ),
        "config": dict(config),
    }
    export.write_output(adapted_graph, stats_export, nested_out_path)

    return (str(args.case), args.current_params, eval_results)


if __name__ == "__main__":
    run()
