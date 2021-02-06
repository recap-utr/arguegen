import csv
import itertools
import logging
import re
import typing as t
from dataclasses import dataclass
from pathlib import Path

import recap_argument_graph as ag
from recap_argument_graph_adaptation.controller.inflect import inflect_concept
from recap_argument_graph_adaptation.model import casebase, query, spacy
from recap_argument_graph_adaptation.model.config import Config
from sklearn.model_selection import ParameterGrid
from spacy.lang.lex_attrs import is_alpha

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


def cases(input_path: Path) -> t.List[casebase.Case]:
    result = []

    for path in itertools.chain([input_path], sorted(input_path.rglob("*"))):
        if path.is_dir() and (case := _case(path, input_path)):
            result.append(case)

    if not result:
        raise RuntimeError(f"No cases were found in '{input_path}'.")

    return result


def _case(path: Path, root_path: Path) -> t.Optional[casebase.Case]:
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

    graph = ag.Graph.from_file(graph_path, casebase.ArgumentNode)
    rules = _parse_rules(rules_path, graph)
    query = _parse_query(query_path)

    return casebase.Case(
        path.relative_to(root_path),
        query,
        graph,
        rules,
    )


def _parse_query(path: Path) -> casebase.UserQuery:
    with path.open() as f:
        text = f.read()

    return casebase.UserQuery(text, spacy.vector(text))


def _parse_rules(path: Path, graph: ag.Graph) -> t.Tuple[casebase.Rule]:
    rules = []

    with path.open() as file:
        reader = csv.reader(file, delimiter=",")

        for row in reader:
            source = _parse_rule_concept(row[0], graph, path, None)
            target = _parse_rule_concept(row[1], graph, path, source.inodes)

            rules.append(casebase.Rule(source, target))

    return tuple(rules)


def _parse_rule_concept(
    rule: str,
    graph: ag.Graph,
    path: Path,
    inodes: t.Optional[t.FrozenSet[casebase.ArgumentNode]],
) -> casebase.Concept:
    rule = rule.strip().lower()
    rule_parts = rule.split("/")
    name = rule_parts[0]
    pos = None

    if len(rule_parts) > 1:
        try:
            pos = casebase.POS(rule_parts[1])
        except ValueError:
            raise ValueError(
                f"The pos '{rule_parts[1]}' specified in '{str(path)}' is invalid."
            )

    kw_name, kw_forms = inflect_concept(name, casebase.pos2spacy(pos))
    vector = spacy.vector(kw_name)

    found_forms = set()

    if not inodes:
        tmp_inodes = set()

        # Only accept rules that cover a complete word.
        # If for example 'landlord' is a rule, but the node only contains 'landlords',
        # an exception will be thrown.
        for kw_form in kw_forms:
            pattern = re.compile(f"\\b({kw_form})\\b")

            for inode in graph.inodes:
                node_txt = inode.plain_text.lower()

                if pattern.search(node_txt):
                    tmp_inodes.add(t.cast(casebase.ArgumentNode, inode))
                    found_forms.add(kw_form)

        if len(tmp_inodes) == 0:
            raise RuntimeError(
                f"The concept '{rule}' with the forms '{kw_forms}' specified in '{str(path)}' could not be found in the graph '{path.parent / str(graph.name)}.json'. Please check the spelling."
            )

        inodes = frozenset(tmp_inodes)
    else:
        found_forms.add(name)

    nodes = query.concept_nodes(
        kw_forms,
        pos,
        [x.vector for x in inodes],
        config["loading"]["min_synset_similarity"],
    )

    if not nodes:
        raise ValueError(
            f"The concept '{rule}' with the forms '{kw_forms}' specified in '{str(path)}' cannot be found in the knowledge graph."
        )

    return casebase.Concept(
        kw_name,
        vector,
        frozenset(found_forms),
        pos,
        inodes,
        nodes,
        {key: 1.0 for key in casebase.metric_keys},
    )
