import csv
import itertools
import logging
import re
import typing as t
from dataclasses import dataclass
from pathlib import Path

import arguebuf as ag
from arguegen.controller.inflect import inflect_concept
from arguegen.model import casebase, nlp, query
from arguegen.model.config import Config
from nltk.corpus import wordnet as wn
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
    param_grid = [params for params in ParameterGrid(dict(config["tuning"]))]
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


def cases(input_path: Path, ignore_errors: bool = False) -> t.List[casebase.Case]:
    result = []

    for path in itertools.chain([input_path], sorted(input_path.rglob("*"))):
        if path.is_dir():
            try:
                case = _case(path, input_path)

                if case:
                    result.append(case)
            except RuntimeError as e:
                if ignore_errors:
                    print(e)
                else:
                    raise e

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

    graph = ag.Graph.from_file(
        graph_path, casebase.HashableAtom, casebase.HashableScheme
    )
    user_query = _parse_query(query_path)
    user_rules = _parse_user_rules(rules_path, graph, user_query)
    rules = (
        user_rules
        if config["loading"]["user_defined_rules"]
        else _generate_system_rules(rules_path, graph, user_query)
    )
    rules_limit = config["loading"]["rule_limit"]
    # TODO: Does not work correctly
    rules_slice = (
        len(rules)
        if config["loading"]["user_defined_rules"] and rules_limit == 0
        else rules_limit
    )

    return casebase.Case(
        path.relative_to(root_path), user_query, graph, rules[:rules_slice], user_rules
    )


def _parse_query(path: Path) -> casebase.UserQuery:
    with path.open() as f:
        text = f.read()

    return casebase.UserQuery(text)


def _parse_user_rules(
    path: Path, graph: ag.Graph, user_query: casebase.UserQuery
) -> t.Tuple[casebase.Rule, ...]:
    rules = []

    with path.open() as file:
        reader = csv.reader(file, delimiter=",")

        for row in reader:
            source_name, source_pos = _split_user_rule(row[0], path)
            target_name, target_pos = _split_user_rule(row[1], path)

            source = _parse_rule_concept(
                source_name, source_pos, graph, user_query, path, None
            )
            target = _parse_rule_concept(
                target_name, target_pos, graph, user_query, path, source.inodes
            )
            rule = _create_rule(source, target, path)

            rules.append(rule)

    _verify_rules(rules, path)

    return tuple(rules)


def _generate_system_rules(
    path: Path, graph: ag.Graph, user_query: casebase.UserQuery
) -> t.Tuple[casebase.Rule, ...]:
    major_claim_nlp, user_query_nlp = nlp.parse_docs(
        [graph.major_claim.text, user_query.text], ["POS"]
    )
    rules = {}

    for source_token, target_token in itertools.product(
        major_claim_nlp, user_query_nlp
    ):
        if (
            source_token.text != target_token.text
            and source_token.pos == target_token.pos
            and source_token.pos_ in config["loading"]["heuristic_pos_tags"]
            and target_token.pos_ in config["loading"]["heuristic_pos_tags"]
        ):
            try:
                source = _parse_rule_concept(
                    source_token.text.strip().lower(),
                    casebase.spacy2pos(source_token.pos_),
                    graph,
                    user_query,
                    path,
                    None,
                )
                target = _parse_rule_concept(
                    target_token.text.strip().lower(),
                    casebase.spacy2pos(target_token.pos_),
                    graph,
                    user_query,
                    path,
                    source.inodes,
                )

                if shortest_paths := query.all_shortest_paths(
                    source.nodes, target.nodes
                ):
                    rule = _create_rule(source, target, path)

                    if distance := len(next(iter(shortest_paths))):
                        rules[rule] = distance

            # If a source or target cannot be found in the knowledge graph,
            # just ignore the error and try the next combination.
            except ValueError:
                pass

    if rules.values():
        min_distance = min(rules.values())

        # Return all rules that have the shortest distance in the knowledge graph.
        return tuple(
            rule for rule, distance in rules.items() if distance == min_distance
        )

    return tuple()


def _verify_rules(rules: t.Collection[casebase.Rule], path: Path) -> None:
    if len(rules) != len({rule.source for rule in rules}):
        raise RuntimeError(
            f"The number of rules specified in '{str(path)}' does not match the number of unique lemmas in the source column. "
            "Please verify that you only specify one rule per lemma (e.g., 'runner/noun'), not one rule per form (e.g., 'runner/noun' and 'runners/noun'). "
            "Different POS tags however should be represented with multiple rules (e.g. 'runner/noun' and 'running/verb')."
        )


def _create_rule(
    source: casebase.Concept, target: casebase.Concept, path: Path
) -> casebase.Rule:
    if config["loading"]["enforce_node_paths"]:
        paths = query.all_shortest_paths(source.nodes, target.nodes)

        if len(paths) == 0:
            err = (
                f"The given rule '{str(source)}->{str(target)}' specified in '{path}' is invalid. "
                "No path to connect the concepts could be found in the knowledge graph. "
            )

            if config["adaptation"]["knowledge_graph"] == "wordnet":
                synsets = [wn.synset(node.uri) for node in source.nodes]
                hypernyms = itertools.chain.from_iterable(
                    hyp.lemmas()
                    for synset in synsets
                    for hyp, _ in synset.hypernym_distances()
                    if not hyp.name().startswith(source.name)
                    and hyp.name() not in config["wordnet"]["hypernym_filter"]
                )
                lemmas = {lemma.name().replace("_", " ") for lemma in hypernyms}

                err += f"The following hypernyms are permitted: {sorted(lemmas)}"

            raise RuntimeError(err)

        source_nodes = frozenset(path.start_node for path in paths)
        target_nodes = frozenset(path.end_node for path in paths)

        source = casebase.Concept.from_concept(source, nodes=source_nodes)
        target = casebase.Concept.from_concept(target, nodes=target_nodes)

    return casebase.Rule(source, target)


def _parse_rule_concept(
    name: str,
    pos: t.Optional[casebase.POS],
    graph: ag.Graph,
    user_query: casebase.UserQuery,
    path: Path,
    inodes: t.Optional[t.FrozenSet[casebase.HashableAtom]],
) -> casebase.Concept:
    kw_name, kw_form2pos, kw_pos2form = inflect_concept(name, casebase.pos2spacy(pos))

    if not inodes:
        tmp_inodes = set()

        # Only accept rules that cover a complete word.
        # If for example 'landlord' is a rule, but the node only contains 'landlords',
        # an exception will be thrown.
        for kw_form in kw_form2pos:
            pattern = re.compile(f"\\b({kw_form})\\b")

            for inode in graph.atom_nodes.values():
                node_txt = inode.plain_text.lower()

                if pattern.search(node_txt):
                    tmp_inodes.add(t.cast(casebase.HashableAtom, inode))

        if not tmp_inodes:
            raise ValueError(
                f"The concept '{name}' with the forms '{kw_form2pos}' specified in '{str(path)}' could not be found in the graph '{path.parent / str(graph.name)}.json'. Please check the spelling."
            )

        inodes = frozenset(tmp_inodes)

    if config["loading"]["filter_kg_nodes"]:
        nodes = query.concept_nodes(kw_form2pos.keys(), pos)
    else:
        nodes = query.concept_nodes(
            kw_form2pos.keys(), pos, [inode.plain_text for inode in inodes]
        )

    if not nodes:
        raise ValueError(
            f"The concept '{name}' with the forms '{kw_form2pos}' specified in '{str(path)}' cannot be found in the knowledge graph."
        )

    return casebase.Concept(
        kw_name,
        kw_form2pos,
        kw_pos2form,
        pos,
        inodes,
        nodes,
        {},
        user_query,
        {key: 1.0 for key in casebase.metric_keys},
    )


def _split_user_rule(rule: str, path: Path) -> t.Tuple[str, casebase.POS]:
    rule = rule.strip().lower()
    rule_parts = rule.split("/")
    name = rule_parts[0]

    if len(rule_parts) > 1:
        try:
            return name, casebase.POS(rule_parts[1])
        except ValueError:
            raise RuntimeError(
                f"The pos '{rule_parts[1]}' specified in '{str(path)}' is invalid."
            )
    else:
        raise RuntimeError(
            f"You did not provide a pos for the rule '{name}' specified in '{str(path)}'."
        )
