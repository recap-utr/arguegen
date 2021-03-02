import logging
import multiprocessing as mp
import typing as t
from pathlib import Path
from timeit import default_timer as timer

import pendulum
import requests
import typer

from recap_argument_graph_adaptation.controller import (
    adapt,
    convert,
    evaluate,
    export,
    extract,
    load,
)
from recap_argument_graph_adaptation.model import casebase, spacy, wordnet
from recap_argument_graph_adaptation.model.config import Config

config = Config.instance()

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
log = logging.getLogger(__name__)


# TODO: Add LaTeX tabular export option.
# TODO: The replacements in the text can be achieved via the inflected forms.
# Each form has a tag associated which can be used to replace it with a matching version.


def _init_child_process():
    # https://stackoverflow.com/a/50379950/7626878
    wordnet.wn = wordnet.init_reader()


def run():
    log.info("Initializing.")
    start_time = timer()
    processes = (
        mp.cpu_count() - 1
        if config["resources"]["processes"] == 0
        else int(config["resources"]["processes"])
    )

    out_path = Path(
        config["resources"]["cases"]["output"],
        pendulum.now().format("YYYY-MM-DD-HH-mm-ss"),
    )
    cases = load.cases(Path(config["resources"]["cases"]["input"]))
    param_grid = load.parameter_grid()
    run_args = load.run_arguments(param_grid, cases, out_path)

    if len(run_args) == 1:
        processes = 1

    log.info(f"Starting with {len(run_args)} runs using {processes} processes.")
    results = []

    if processes == 1:
        logging.getLogger(__package__).setLevel(logging.DEBUG)
        results = [_parametrized_run(run_arg) for run_arg in run_args]
    else:
        with mp.Pool(processes, initializer=_init_child_process) as pool:
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

    if config["export"]["grid_stats"] and len(run_args) > 1:
        export.grid_stats(results, duration, param_grid, out_path)

    log.info(f"Finished in {duration} sec.")
    log.info(f"Average duration per run: {duration / len(run_args)}")


def _parametrized_run(args: load.RunArgs) -> t.Tuple[str, int, casebase.Evaluation]:
    log.debug(f"Starting run {args.current_run + 1}/{args.total_runs}.")

    config["_tuning"] = args.params
    spacy.session = requests.Session()
    case = args.case

    log.debug("Starting adaptation pipeline.")
    start_time = timer()
    nested_out_path = export.nested_path(
        args.out_path / case.relative_path, args.total_params, config["_tuning"]
    )

    adapted_concepts = {}
    reference_paths = {}
    adapted_paths = {}
    adapted_concept_candidates = {}
    adapted_graph = None

    log.debug("Extracting keywords.")
    relevant_concepts, all_concepts = extract.keywords(
        case.graph, case.rules, case.user_query
    )

    log.debug("Adapting concepts.")
    adaptation_method = config.tuning("adaptation", "method")

    if adaptation_method == "direct":
        adapted_concepts, adapted_concept_candidates = adapt.concepts(
            relevant_concepts, case.rules, args.case.user_query
        )

    elif adaptation_method == "bfs":
        reference_paths = extract.paths(relevant_concepts, case.rules)
        adapted_concepts, adapted_paths, adapted_concept_candidates = adapt.paths(
            reference_paths, case.rules, args.case.user_query
        )

    if config["export"]["graph"]:
        log.debug("Exporting graph.")
        adapted_graph = adapt.argument_graph(case.graph, case.rules, adapted_concepts)

    log.debug("Evaluating adaptations.")
    eval_results = evaluate.case(case, adapted_concepts, all_concepts)

    end_time = timer()

    log.debug("Exporting statistics.")
    stats_export = {
        "path": str(Path(config["resources"]["cases"]["input"]) / case.relative_path),
        "evaluation": eval_results.to_dict(),
        "time": end_time - start_time,
        "rules": {
            "benchmark_rules": convert.list_str(case.benchmark_rules),
            "case_rules": convert.list_str(case.rules),
        },
        "results": export.statistic(
            relevant_concepts,
            reference_paths,
            adapted_paths,
            adapted_concept_candidates,
            adapted_concepts,
        ),
        "config": dict(config),
    }
    export.write_output(adapted_graph, stats_export, nested_out_path)

    return (str(args.case), args.current_params, eval_results)


if __name__ == "__main__":
    run()
