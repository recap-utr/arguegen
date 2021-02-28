import json
import logging
import shutil
import statistics
import typing as t
from collections import defaultdict
from pathlib import Path

import recap_argument_graph as ag
from recap_argument_graph_adaptation.controller import convert
from recap_argument_graph_adaptation.model import casebase, graph
from recap_argument_graph_adaptation.model.config import Config

log = logging.getLogger(__name__)
config = Config.instance()


def statistic(
    concepts: t.Iterable[casebase.Concept],
    reference_paths: t.Mapping[casebase.Concept, t.Iterable[graph.AbstractPath]],
    adapted_paths: t.Mapping[casebase.Concept, t.Iterable[graph.AbstractPath]],
    adapted_concept_candidates: t.Mapping[
        casebase.Concept, t.Iterable[casebase.Concept]
    ],
    adapted_concepts: t.Mapping[casebase.Concept, casebase.Concept],
) -> t.Dict[str, t.Any]:
    out = {}

    for concept in concepts:
        key = f"({concept})->({adapted_concepts.get(concept)})"
        candidates_str = None

        if candidates_for_concept := adapted_concept_candidates.get(concept):
            candidates_str = {
                str(item): item.score
                for item in sorted(candidates_for_concept, key=lambda x: x.score)
            }

        out[key] = {
            **concept.to_dict(),
            "reference_paths": convert.list_str(reference_paths.get(concept)),
            "adapted_paths": convert.list_str(adapted_paths.get(concept)),
            "adaptation_candidates": candidates_str,
            "adapted_name": convert.xstr(adapted_concepts.get(concept)),
        }

    return out


def grid_stats(
    results: t.Iterable[t.Tuple[str, int, casebase.Evaluation]],
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
            current_path = nested_path(
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
                current_path = nested_path(
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
        "global_config": dict(config),
    }

    grid_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with grid_stats_path.open("w") as file:
        _json_dump(grid_stats, file)

    log.info(f"Grid stats exported to {str(grid_stats_path)}")


def nested_path(
    path: Path, total_runs: int, nested_folders: t.Mapping[str, t.Any]
) -> Path:
    nested_path = path

    if total_runs > 1:
        for tuning_key, tuning_value in nested_folders.items():
            nested_path /= f"{tuning_key}_{tuning_value}"

    return nested_path


def write_output(
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
        # sort_keys=True,
    )


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


def _file_path(path: Path) -> str:
    return "file://" + str(path)
