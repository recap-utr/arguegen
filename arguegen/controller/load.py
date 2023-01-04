import csv
import itertools
import logging
import re
import typing as t
from dataclasses import dataclass
from pathlib import Path

import arguebuf as ag
from sklearn.model_selection import ParameterGrid

from arguegen.config import config
from arguegen.controller.inflect import inflect_concept
from arguegen.model import casebase, nlp, wordnet

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


def cases(
    input_path: Path, ignore_errors: bool = False, allow_empty_rules: bool = False
) -> t.List[casebase.Case]:
    result = []

    for path in itertools.chain([input_path], sorted(input_path.rglob("*"))):
        if path.is_dir():
            try:
                case = _case(path, input_path, allow_empty_rules)

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


def _case(
    path: Path, root_path: Path, allow_empty_rules: bool
) -> t.Optional[casebase.Case]:
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
        graph_path,
        atom_class=casebase.HashableAtom,
        scheme_class=casebase.HashableScheme,
    )
    user_query = _parse_query(query_path)
    user_rules = _parse_user_rules(rules_path, graph, user_query)
    rules = (
        user_rules
        if config["loading"]["user_defined_rules"]
        else _generate_system_rules(rules_path, graph, user_query)
    )

    log.debug(
        f"Using the following {'user' if config.loading.user_defined_rules else 'system'}-generated rules: "
        + ", ".join(str(rule) for rule in rules)
    )

    if not user_rules:
        raise RuntimeError(f"No rules found in file '{rules_path}'.")
    if not rules and not allow_empty_rules:
        return None

    user_rules_limit = config["loading"]["user_rules_limit"]
    rules_slice = (
        user_rules_limit
        if config["loading"]["user_defined_rules"] and user_rules_limit > 0
        else len(rules)
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
                target_name, target_pos, graph, user_query, path, source.atoms
            )
            rule = _create_rule(source, target, path)

            rules.append(rule)

    _verify_rules(rules, path)

    return tuple(rules)


def _generate_system_rules(
    path: Path, graph: ag.Graph, user_query: casebase.UserQuery
) -> t.Tuple[casebase.Rule, ...]:
    mc = graph.major_claim or graph.root_node

    if not mc:
        return tuple()

    rules: dict[casebase.Rule, int] = {}
    major_claim_kw = nlp.keywords([mc.text], config["loading"]["heuristic_pos_tags"])
    user_query_kw = nlp.keywords(
        [user_query.text], config["loading"]["heuristic_pos_tags"]
    )

    for source_token, target_token in itertools.product(
        major_claim_kw,
        user_query_kw,
    ):
        if (
            source_token.lemma != target_token.lemma
            and source_token.pos_tag == target_token.pos_tag
        ):
            try:
                source = _parse_rule_concept(
                    source_token.lemma.strip(),
                    casebase.spacy2pos(source_token.pos_tag),
                    graph,
                    user_query,
                    path,
                    None,
                )
                target = _parse_rule_concept(
                    target_token.lemma.strip(),
                    casebase.spacy2pos(target_token.pos_tag),
                    graph,
                    user_query,
                    path,
                    source.atoms,
                )

                if shortest_paths := wordnet.all_shortest_paths(
                    source.synsets, target.synsets
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
            rule
            for rule, distance in rules.items()
            if distance == min_distance and rule.source != rule.target
        )

    return tuple()

    # TODO: be is adapted to be
    # microtexts-premtim/keep_retirement_at_63/nodeset6416
    # generated: ["(be/verb/{'120420', '120419', '120421', '120418'})->(be/verb/{'120420', '120419', '120421', '120418'})", "(economy/noun/{'120421'})->(system/noun/{'120421'})"]
    # user: ["(retirement/noun/{'120418'})->(termination/noun/{'120418'})", "(economy/noun/{'120421'})->(system/noun/{'120421'})", "(capability/noun/{'120420'})->(ability/noun/{'120420'})", "(labour/noun/{'120421'})->(work/noun/{'120421'})"]

    # TODO: Noun chunks shoudl also be considered. At the moment, 'morning-after pill' is ignored and only 'pill' is considered


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
        paths = wordnet.all_shortest_paths(source.synsets, target.synsets)

        if len(paths) == 0:
            err = (
                f"The given rule '{str(source)}->{str(target)}' specified in '{path}' is invalid. "
                "No path to connect the concepts could be found in the knowledge graph. "
            )

            lemmas = itertools.chain.from_iterable(
                hyp.lemmas
                for synset in source.synsets
                for hyp, _ in synset.hypernym_distances().items()
                if not any(lemma.startswith(source.lemma) for lemma in hyp.lemmas)
                and hyp._synset.id not in config["wordnet"]["hypernym_filter"]
            )

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
    atoms: t.Optional[t.FrozenSet[casebase.HashableAtom]],
) -> casebase.Concept:
    lemma, form2pos, pos2form = inflect_concept(
        name, casebase.pos2spacy(pos), lemmatize=True
    )

    if not atoms:
        tmp_atoms = set()

        # Only accept rules that cover a complete word.
        # If for example 'landlord' is a rule, but the node only contains 'landlords',
        # an exception will be thrown.
        for form in form2pos:
            pattern = re.compile(f"\\b({form})\\b")

            for atom in graph.atom_nodes.values():
                atom_txt = atom.plain_text.lower()

                if pattern.search(atom_txt):
                    tmp_atoms.add(t.cast(casebase.HashableAtom, atom))

        if not tmp_atoms:
            raise ValueError(
                f"The concept '{name}' with the forms '{form2pos}' specified in '{str(path)}' could not be found in the graph '{path.parent / str(graph.name)}.json'. Please check the spelling."
            )

        atoms = frozenset(tmp_atoms)

    if not config["loading"]["filter_kg_nodes"]:
        synsets = wordnet.concept_synsets(form2pos.keys(), pos)
    else:
        synsets = wordnet.concept_synsets(
            form2pos.keys(), pos, [atom.plain_text for atom in atoms]
        )

    if not synsets:
        raise ValueError(
            f"The concept '{name}' with the forms '{form2pos}' specified in '{str(path)}' cannot be found in the knowledge graph."
        )

    return casebase.Concept(
        lemma,
        form2pos,
        pos2form,
        pos,
        atoms,
        synsets,
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
