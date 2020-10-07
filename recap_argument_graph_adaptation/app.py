import json
import logging
import os
import typing as t
from pathlib import Path

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

    for case in cases:
        adaptation_methods = [adaptation.Method(config["adaptation"]["method"])]
        adaptation_selectors = [adaptation.Selector(config["adaptation"]["selector"])]

        if config["adaptation"]["gridsearch"]:
            adaptation_methods = [method for method in adaptation.Method]
            adaptation_selectors = [selector for selector in adaptation.Selector]

        for adaptation_method in adaptation_methods:
            for adaptation_selector in adaptation_selectors:
                _perform_adaptation(
                    case, adaptation_method, adaptation_selector, out_path
                )


def _perform_adaptation(
    case: adaptation.Case,
    adaptation_method: adaptation.Method,
    adaptation_selector: adaptation.Selector,
    out_path: Path,
) -> None:
    log.info(
        f"Processing '{case.name}' "
        f"with method '{adaptation_method.value}' "
        f"and selector '{adaptation_selector.value}'."
    )

    nested_out_path: Path = out_path / case.name

    if config["adaptation"]["gridsearch"]:
        nested_out_path = (
            nested_out_path / adaptation_method.value / adaptation_selector.value
        )

    nested_out_path.mkdir(parents=True, exist_ok=True)

    adaptation_results = {}

    for rule in case.rules:
        log.info(f"Processing rule {str(rule)}.")

        concepts = extract.keywords(case.graph, rule)
        reference_paths = extract.paths(concepts, rule, adaptation_method)
        adapted_concepts, adapted_paths = adapt.paths(
            reference_paths, rule, adaptation_selector, adaptation_method
        )
        adapt.argument_graph(case.graph, rule, adapted_concepts)

        adaptation_results[str(rule)] = export.statistic(
            concepts, reference_paths, adapted_concepts, adapted_paths
        )

    stats = {"results": adaptation_results, "config": dict(config)}

    _write_output(case, stats, nested_out_path)


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
