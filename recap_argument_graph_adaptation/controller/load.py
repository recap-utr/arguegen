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
        if round(sum(Config.filter_mapping(params, "weight").values()), 2) == 1
        and round(sum(Config.filter_mapping(params, "score").values()), 2) == 1
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


def cases() -> t.List[casebase.Case]:
    input_path = Path(config["resources"]["cases"]["input"])
    result = []

    for folder in sorted(input_path.iterdir()):
        if folder.is_dir():
            result.append(_case(folder))

    if not result:  # no nested folders were found
        result.append(_case(input_path))

    return result


def _case(path: Path) -> casebase.Case:
    name = path.name
    graph = ag.Graph.open(path / "graph.json")
    rules = _parse_rules(path / "rules.csv")
    query = _parse_txt(path / "query.txt")

    if not (graph and rules and query):
        raise RuntimeError(
            "Not all required assets ('graph.json', 'rules.csv', 'query.txt') were found"
        )

    return casebase.Case(
        name,
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
            source = _parse_rule_concept(row[0])
            target = _parse_rule_concept(row[1])

            rules.append(casebase.Rule(source, target))

    return tuple(rules)


def _parse_rule_concept(rule: str) -> casebase.Concept:
    rule_parts = rule.split("/")
    name = rule_parts[0]
    vector = spacy.vector(name)
    pos = None

    if len(rule_parts) > 1:
        pos = casebase.POS(rule_parts[1])

    nodes = query.concept_nodes(name, pos)

    if not nodes:
        raise ValueError(
            f"The rule concept '{name}' cannot be found in the knowledge graph."
        )

    return casebase.Concept(
        name,
        vector,
        pos,
        nodes,
        {key: 1.0 for key in casebase.metric_keys},
    )
