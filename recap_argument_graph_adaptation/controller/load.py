import csv
import typing as t
from pathlib import Path

import recap_argument_graph as ag
from recap_argument_graph_adaptation.model import adaptation
from recap_argument_graph_adaptation.model.config import config


def cases() -> t.List[adaptation.Case]:
    case_folder = Path(config["path"]["input"])
    result = []

    for graph_file in case_folder.rglob("*.json"):
        graph = ag.Graph.open(graph_file)
        rule_file = graph_file.with_suffix(".csv")

        if rule_file.is_file():
            rules = _parse_rules(rule_file)

            result.append(adaptation.Case(graph=graph, rules=rules))

    return result


def _parse_rules(path: Path) -> t.List[adaptation.Rule]:
    rules = []

    with path.open() as file:
        reader = csv.reader(file, delimiter=",")

        for row in reader:
            rules.append((row[0], row[1]))

    return rules
