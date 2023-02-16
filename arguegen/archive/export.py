import json
import logging
import shutil
import typing as t
from collections import defaultdict
from pathlib import Path

import arguebuf as ag

from arguegen.config import config
from arguegen.controllers import convert
from arguegen.model import casebase, evaluation, wordnet

log = logging.getLogger(__name__)


def statistic(
    concepts: t.Iterable[casebase.Concept],
    reference_paths: t.Mapping[casebase.Concept, t.Iterable[wordnet.Path]],
    adapted_paths: t.Mapping[casebase.Concept, t.Iterable[wordnet.Path]],
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
    results: t.Iterable[t.Tuple[str, int, evaluation.EvaluationTuple]],
    duration: float,
    param_grid: t.Sequence[t.Mapping[str, t.Any]],
    out_path: Path,
) -> None:
    log.info("Exporting grid stats.")

    results = [entry for entry in results if entry is not None]
    case_results: t.Dict[str, t.List[t.Tuple[evaluation.EvaluationTuple, int]]] = (
        defaultdict(list)
    )
    param_combinations: t.List[t.List[evaluation.EvaluationTuple]] = [
        [] for _ in range(len(param_grid))
    ]
    param_results: t.Dict[str, t.Dict[str, t.List[evaluation.EvaluationTuple]]] = {
        key: defaultdict(list) for key in config["tuning"]
    }
    score_distribution = []

    for case, i, eval in results:
        case_results[case].append((eval, i))
        param_combinations[i].append(eval)
        score_distribution.append(eval)

        for key in config["tuning"].keys():
            param_results[key][str(param_grid[i][key])].append(eval)

    best_case_results = {}

    for case, eval_tuples in case_results.items():
        eval_tuples.sort(key=lambda x: x[0].synthesis.score, reverse=True)
        _results = []

        for eval_tuple, i in eval_tuples:
            current_path = nested_path(out_path / case, len(param_grid), param_grid[i])

            # Move the best results to the root folder for that case.
            if len(_results) == 0:
                copy_case_files(current_path, out_path / case, "best")
            elif len(_results) == len(eval_tuples) - 1:
                copy_case_files(current_path, out_path / case, "worst")
            elif len(_results) == len(eval_tuples) // 2:
                copy_case_files(current_path, out_path / case, "median")

            _results.append(
                {
                    "evaluation": eval_tuple.to_dict(compact=True),
                    "files": _output_file_paths(current_path),
                    "config": param_grid[i],
                }
            )

        best_case_results[case] = _results

    mean_param_combinations = []

    for i, eval_tuples in enumerate(param_combinations):
        if eval_tuples:
            current_cases = {}

            for case, case_eval_tuples in case_results.items():
                case_eval_tuple, _ = next(
                    filter(lambda x: x[1] == i, case_eval_tuples), (None, None)
                )
                current_path = nested_path(
                    out_path / case, len(param_grid), param_grid[i]
                )
                current_cases[case] = {
                    "evaluation": (
                        case_eval_tuple.to_dict(compact=True)
                        if case_eval_tuple
                        else None
                    ),
                    "files": _output_file_paths(current_path),
                }

            eval_results_aggr = evaluation.aggregate_eval(eval_tuples)

            mean_param_combinations.append(
                {
                    "evaluation": eval_results_aggr,
                    "config": param_grid[i],
                    "cases": current_cases,
                }
            )

    mean_param_combinations.sort(
        key=_param_sorter,
        reverse=True,
    )

    mean_param_results = {}

    # https://stackoverflow.com/a/33046935
    for param_key, param_values in param_results.items():
        mean_param_results[param_key] = {}

        for param_value, eval_results in param_values.items():
            mean_param_results[param_key][param_value] = evaluation.aggregate_eval(
                eval_results
            )

    grid_stats_path = out_path / "grid_stats.json"
    grid_stats = {
        "duration": duration,
        "score_distribution": evaluation.aggregate_eval(score_distribution),
        "param_results": mean_param_results,
        "param_combinations": mean_param_combinations,
        "case_results": best_case_results,
        "global_config": dict(config),
    }

    grid_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with grid_stats_path.open("w") as file:
        _json_dump(grid_stats, file)

    log.info(f"Grid stats exported to {str(grid_stats_path)}")


def _param_sorter(x: t.Mapping[str, t.Any]) -> float:
    sort_key = config["export"]["sort_key"]

    if (val := x["evaluation"]["synthesis"].get(sort_key)) is not None:
        return val

    return x["evaluation"]["synthesis"]["mean"][sort_key]


def nested_path(
    path: Path, total_runs: int, nested_folders: t.Mapping[str, t.Any]
) -> Path:
    nested_path = path

    if total_runs > 1:
        for tuning_key, tuning_value in nested_folders.items():
            if isinstance(tuning_value, list):
                tuning_value = "_".join(tuning_value)

            nested_path /= f"{tuning_key}_{tuning_value}"

    return nested_path


def copy_case_files(source_path: Path, destination_path: Path, prefix: str) -> None:
    for file in ("case.json", "case.pdf", "stats.json"):
        try:
            shutil.copy(source_path / file, destination_path / f"{prefix}_{file}")
        except Exception:
            pass


def write_output(
    adapted_graph: t.Optional[ag.Graph], stats: t.Mapping[str, t.Any], path: Path
) -> None:
    path.mkdir(parents=True, exist_ok=True)

    if adapted_graph:
        if config["export"]["graph_json"]:
            adapted_graph.to_file(path / "case.json", format=ag.GraphFormat.ARGUEBUF)

        if config["export"]["graph_pdf"]:
            ag.render(adapted_graph.to_gv(), path / "case.pdf")

    if config["export"]["single_stats"]:
        stats_path = path / "stats.json"

        with stats_path.open("w") as file:
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

    if config["export"]["graph_json"]:
        filenames.append("case.json")

    if config["export"]["graph_pdf"]:
        filenames.append("case.pdf")

    if config["export"]["single_stats"]:
        filenames.append("stats.json")

        for filename in filenames:
            paths[filename] = _file_path(parent_folder / filename)

    return paths


def _file_path(path: Path) -> str:
    if config["resources"]["relative_paths"]:
        return str(path)

    return "file://" + str(path.resolve())
