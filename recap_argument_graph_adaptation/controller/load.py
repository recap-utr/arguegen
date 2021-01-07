import csv
import logging
import typing as t
from pathlib import Path

import recap_argument_graph as ag
from recap_argument_graph_adaptation.controller import metrics, spacy, wordnet
from recap_argument_graph_adaptation.model import adaptation, graph
from recap_argument_graph_adaptation.model.config import config
from recap_argument_graph_adaptation.model.database import Database

log = logging.getLogger(__name__)


def cases() -> t.List[adaptation.Case]:
    input_path = Path(config["resources"]["cases"]["input"])
    result = []

    for folder in sorted(input_path.iterdir()):
        if folder.is_dir():
            result.append(_case(folder))

    if not result:  # no nested folders were found
        result.append(_case(input_path))

    return result


# TODO: Output case and benchmark rules for verification purposes.


def _case(path: Path) -> adaptation.Case:
    name = path.name
    graph = ag.Graph.open(path / "graph.json")
    rules = _parse_rules(path / "rules.csv")
    query = _parse_txt(path / "query.txt")

    if not (graph and rules and query):
        raise RuntimeError(
            "Not all required assets ('graph.json', 'rules.csv', 'query.txt') were found"
        )

    return adaptation.Case(
        name,
        query,
        graph,
        rules,
    )


def _parse_txt(path: Path) -> str:
    with path.open() as f:
        return f.read()


def _parse_rules(path: Path) -> t.Tuple[adaptation.Rule]:
    rules = []

    with path.open() as file:
        reader = csv.reader(file, delimiter=",")

        for row in reader:
            source = _parse_rule_concept(row[0])
            target = _parse_rule_concept(row[1])

            rules.append(adaptation.Rule(source, target))

    return tuple(rules)


def _parse_rule_concept(rule: str) -> adaptation.Concept:
    rule_parts = rule.split("/")
    name = rule_parts[0]
    vector = spacy.vector(name)
    pos = graph.POS.OTHER

    if len(rule_parts) > 1:
        pos = graph.POS(rule_parts[1])

    db = Database()
    nodes = db.nodes(name, pos)
    synsets = wordnet.concept_synsets(name, pos)

    if config["adaptation"]["knowledge_graph"] == "conceptnet" and not nodes:
        raise ValueError(f"The rule concept '{name}' cannot be found in ConceptNet.")
    elif config["adaptation"]["knowledge_graph"] == "wordnet" and not synsets:
        raise ValueError(
            f"The rule concept '{name}/{pos.value}' cannot be found in WordNet."
        )

    return adaptation.Concept(
        name, vector, pos, nodes, tuple(synsets), None, *metrics.best_concept_metrics
    )
