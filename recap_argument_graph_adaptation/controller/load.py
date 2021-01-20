import csv
import itertools
import logging
import typing as t
from dataclasses import dataclass
from pathlib import Path

import recap_argument_graph as ag
from recap_argument_graph_adaptation.model import casebase, query, spacy
from recap_argument_graph_adaptation.model.config import Config
from sklearn.model_selection import ParameterGrid

config = Config.instance()
log = logging.getLogger(__name__)


@dataclass()
class RunArgs:
    current_run: int
    total_runs: int
    current_params: int
    total_params: int
    params: t.Mapping[str, t.Any]
    current_case: int
    total_cases: int
    case: casebase.Case
    out_path: Path


def parameter_grid() -> t.List[t.Dict[str, t.Any]]:
    param_grid = [
        params
        for params in ParameterGrid(dict(config["tuning"]))
        # if round(sum(Config.filter_mapping(params, "weight").values()), 2) == 1
        # and round(sum(Config.filter_mapping(params, "score").values()), 2) == 1
    ]
    if not param_grid:
        param_grid = [{key: values[0] for key, values in config["tuning"].items()}]

    return param_grid


def run_arguments(
    param_grid: t.Collection[t.Mapping[str, t.Any]],
    cases: t.Collection[casebase.Case],
    out_path: Path,
):
    total_params = len(param_grid)
    total_cases = len(cases)
    total_runs = total_params * total_cases

    return [
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


input_path = Path(config["resources"]["cases"]["input"])


def cases() -> t.List[casebase.Case]:
    result = []

    for path in itertools.chain([input_path], sorted(input_path.rglob("*"))):
        if path.is_dir() and (case := _case(path)):
            result.append(case)

    if not result:
        raise RuntimeError(f"No cases were found in '{input_path}'.")

    return result


def _case(path: Path) -> t.Optional[casebase.Case]:
    graph_path = path / "graph.json"
    rules_path = path / "rules.csv"
    query_path = path / "query.txt"
    paths = (graph_path, rules_path, query_path)

    # If the folder does not contain any file, ignore it.
    if not any(p.exists() for p in paths):
        return None

    # If some files exist, but not all, raise an exception.
    if not all(p.exists() for p in paths):
        raise RuntimeError(
            f"Only some of the required assets {[p.name for p in paths]} were found in '{path}'."
        )

    graph = ag.Graph.open(graph_path)
    rules = _parse_rules(rules_path)
    query = _parse_txt(query_path)

    return casebase.Case(
        path.relative_to(input_path),
        query,
        graph,
        rules,
    )


def _parse_txt(path: Path) -> str:
    with path.open() as f:
        return f.read()


def _parse_rules(path: Path) -> t.Tuple[casebase.Rule]:
    rules = []

    with path.open() as file:
        reader = csv.reader(file, delimiter=",")

        for row in reader:
            source = _parse_rule_concept(row[0], path)
            target = _parse_rule_concept(row[1], path)

            rules.append(casebase.Rule(source, target))

    return tuple(rules)


def _parse_rule_concept(rule: str, path: Path) -> casebase.Concept:
    rule_parts = rule.split("/")
    name = rule_parts[0]
    vector = spacy.vector(name)
    pos = None

    if len(rule_parts) > 1:
        try:
            pos = casebase.POS(rule_parts[1])
        except ValueError:
            raise ValueError(
                f"The pos '{rule_parts[1]}' specified in '{str(path)}' is invalid."
            )

    nodes = query.concept_nodes(name, pos)

    if not nodes:
        raise ValueError(
            f"The concept '{rule}' specified in '{str(path)}' cannot be found in the knowledge graph."
        )

    return casebase.Concept(
        name,
        vector,
        pos,
        nodes,
        {key: 1.0 for key in casebase.metric_keys},
    )
