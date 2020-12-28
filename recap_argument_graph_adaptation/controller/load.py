import csv

import numpy as np
from recap_argument_graph_adaptation.controller import metrics, spacy, wordnet
import typing as t
from pathlib import Path

import recap_argument_graph as ag
from recap_argument_graph_adaptation.model import adaptation, graph
from recap_argument_graph_adaptation.model.config import config
from recap_argument_graph_adaptation.model.database import Database


def cases() -> t.List[adaptation.Case]:
    input_path = Path(config["path"]["input"])
    result = []

    for folder in sorted(input_path.iterdir()):
        if folder.is_dir():
            result.append(_case(folder))

    if not result:  # no nested folders were found
        result.append(_case(input_path))

    return result


def _case(path: Path) -> adaptation.Case:
    input_graph = ag.Graph.open(path / "case-graph.json")
    input_rules = _parse_rules(path / "case-rules.csv")

    benchmark_graph = ag.Graph.open(path / "benchmark-graph.json")
    benchmark_rules = _parse_rules(path / "benchmark-rules.csv")

    if not (input_graph and input_rules and benchmark_graph and benchmark_rules):
        raise RuntimeError(
            "Not all required assets ('case-graph.json', 'case-rules.csv', 'benchmark-graph.json', 'benchmark-rules.csv') were found"
        )

    with (path / "query.txt").open() as file:
        query = file.read()

    return adaptation.Case(
        path.name,
        query,
        input_graph,
        input_rules,
        benchmark_graph,
        benchmark_rules,
    )


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
    synsets = wordnet.synsets(name, pos)

    if config["nlp"]["knowledge_graph"] == "conceptnet" and not nodes:
        raise ValueError(f"The rule concept '{name}' cannot be found in ConceptNet.")
    elif config["nlp"]["knowledge_graph"] == "wordnet" and not synsets:
        raise ValueError(
            f"The rule concept '{name}/{pos.value}' cannot be found in WordNet."
        )

    return adaptation.Concept(
        name, vector, pos, nodes, synsets, None, *metrics.best_concept_metrics
    )
