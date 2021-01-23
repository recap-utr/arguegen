import itertools
import logging
import re
import typing as t
from collections import defaultdict
from dataclasses import dataclass

import recap_argument_graph as ag
from recap_argument_graph_adaptation.model import casebase, graph, query, spacy
from recap_argument_graph_adaptation.model.config import Config

config = Config.instance()

log = logging.getLogger(__name__)


def argument_graph(
    original_graph: ag.Graph,
    rules: t.Collection[casebase.Rule],
    adapted_concepts: t.Mapping[casebase.Concept, casebase.Concept],
) -> ag.Graph:
    substitutions = {
        concept.name: adapted_concept.name
        for concept, adapted_concept in adapted_concepts.items()
    }
    for rule in rules:
        substitutions[rule.source.name] = rule.target.name

    adapted_graph = original_graph.copy()

    for node in adapted_graph.inodes:
        node.text = _replace(node.plain_text, substitutions)
        # node.text = pr.proofread(node.text)

    return adapted_graph


def _replace(text: str, substitutions: t.Mapping[str, str]):
    substrings = sorted(substitutions.keys(), key=len, reverse=True)

    for substring in substrings:
        pattern = re.compile(substring, re.IGNORECASE)
        text = pattern.sub(substitutions[substring], text)

    return text


def concepts(
    concepts: t.Iterable[casebase.Concept],
    rules: t.Collection[casebase.Rule],
    user_query: casebase.UserQuery,
) -> t.Tuple[
    t.Dict[casebase.Concept, casebase.Concept],
    t.Dict[casebase.Concept, t.Set[casebase.Concept]],
]:
    all_candidates = {}
    adapted_concepts = {}
    related_concept_weight = config.tuning("weight")

    for original_concept in concepts:
        adaptation_candidates = set()
        related_concepts = {
            original_concept: related_concept_weight["original_concept"]
        }

        for rule in rules:
            related_concepts.update(
                {
                    rule.target: related_concept_weight["rule_target"] / len(rules),
                    rule.source: related_concept_weight["rule_source"] / len(rules),
                }
            )

        for node in original_concept.nodes:
            hypernym_distances = node.hypernym_distances(
                original_concept.inode_vectors,
                config.tuning("adaptation", "min_synset_similarity"),
            )

            for hypernym, hyp_distance in hypernym_distances.items():
                name = hypernym.processed_name
                pos = hypernym.pos
                vector = spacy.vector(name)
                nodes = frozenset([hypernym])

                candidate = casebase.Concept(
                    name,
                    vector,
                    query.pos(pos),
                    original_concept.inodes,
                    nodes,
                    query.concept_metrics(
                        related_concepts,
                        user_query,
                        nodes,
                        vector,
                        hypernym_level=hyp_distance,
                    ),
                )

                adaptation_candidates.add(candidate)

        all_candidates[original_concept] = adaptation_candidates
        adapted_concept = _filter_concepts(
            adaptation_candidates, original_concept, rules
        )

        if adapted_concept:
            adapted_concepts[original_concept] = adapted_concept
            log.debug(f"Adapt ({original_concept})->({adapted_concept}).")

        else:
            log.debug(f"No adaptation for ({original_concept}).")

    return adapted_concepts, all_candidates


def paths(
    reference_paths: t.Mapping[casebase.Concept, t.Sequence[graph.AbstractPath]],
    rules: t.Collection[casebase.Rule],
    user_query: casebase.UserQuery,
) -> t.Tuple[
    t.Dict[casebase.Concept, casebase.Concept],
    t.Dict[casebase.Concept, t.List[graph.AbstractPath]],
    t.Dict[casebase.Concept, t.Set[casebase.Concept]],
]:
    related_concept_weight = config.tuning("weight")
    adapted_concepts = {}
    adapted_paths = {}
    all_candidates = {}

    for original_concept, all_shortest_paths in reference_paths.items():
        adaptation_results = [
            _bfs_adaptation(shortest_path, original_concept, rule)
            for shortest_path, rule in itertools.product(all_shortest_paths, rules)
        ]

        adaptation_results = list(itertools.chain(*adaptation_results))
        adapted_paths[original_concept] = adaptation_results
        _adaptation_candidates = defaultdict(list)

        for result in adaptation_results:
            hyp_distance = len(result)
            name = result.end_node.processed_name
            vector = spacy.vector(name)
            end_nodes = frozenset([result.end_node])
            pos = query.pos(result.end_node.pos)
            related_concepts = {}

            for rule in rules:
                related_concepts.update(
                    {
                        rule.target: related_concept_weight["rule_target"] / len(rules),
                        rule.source: related_concept_weight["rule_source"] / len(rules),
                        original_concept: related_concept_weight["original_concept"]
                        / len(rules),
                    }
                )

            candidate = casebase.Concept(
                name,
                vector,
                pos,
                original_concept.inodes,
                end_nodes,
                query.concept_metrics(
                    related_concepts,
                    user_query,
                    end_nodes,
                    vector,
                    hypernym_level=hyp_distance,
                ),
            )

            _adaptation_candidates[str(candidate)].append(candidate)

        adaptation_candidates = set()
        candidate_occurences = {}
        candidate_length_diff = defaultdict(lambda: float("inf"))

        for candidates in _adaptation_candidates.values():
            candidate = max(candidates, key=lambda x: x.score)

            adaptation_candidates.add(candidate)
            candidate_occurences[candidate] = len(candidates)
            candidate_length_diff[candidate] = abs(
                len(candidate.nodes) - len(all_shortest_paths[0].nodes)
            )

        all_candidates[original_concept] = adaptation_candidates

        adapted_concept = _filter_concepts(
            adaptation_candidates,
            original_concept,
            rules,
            candidate_occurences,
            candidate_length_diff,
        )

        if adapted_concept:
            adapted_concepts[original_concept] = adapted_concept
            log.debug(f"Adapt ({original_concept})->({adapted_concept}).")

        else:
            log.debug(f"No adaptation for ({original_concept}).")

    return adapted_concepts, adapted_paths, all_candidates


def _bfs_adaptation(
    shortest_path: graph.AbstractPath,
    concept: casebase.Concept,
    rule: casebase.Rule,
) -> t.Set[graph.AbstractPath]:
    method = config.tuning("bfs", "method")
    adapted_paths = set()

    # We have to convert the target to a path object here.
    start_nodes = rule.target.nodes if method == "within" else concept.nodes

    for start_node in start_nodes:
        current_paths = set()

        current_paths.add(
            graph.AbstractPath.from_node(start_node)
        )  # Start with only one node.

        for _ in shortest_path.relationships:
            next_paths = set()

            for current_path in current_paths:
                path_extensions = query.hypernyms_as_paths(
                    current_path.end_node,
                    concept.inode_vectors,
                    config.tuning("adaptation", "min_synset_similarity"),
                )

                # Here, paths that are shorter than the reference path are discarded.
                if path_extensions:
                    path_extensions = _filter_paths(
                        current_path, path_extensions, shortest_path
                    )

                    for path_extension in path_extensions:
                        next_paths.add(
                            graph.AbstractPath.merge(current_path, path_extension)
                        )

                # We still want these shorter paths, they can be filtered later
                else:
                    adapted_paths.add(current_path)

            current_paths = next_paths

        adapted_paths.update(current_paths)

    return adapted_paths


def _dist2sim(distance: t.Optional[float]) -> t.Optional[float]:
    if distance is not None:
        return 1 / (1 + distance)

    return None


def _filter_concepts(
    concepts: t.Set[casebase.Concept],
    original_concept: casebase.Concept,
    rules: t.Collection[casebase.Rule],
    occurences: t.Optional[t.Mapping[casebase.Concept, int]] = None,
    length_differences: t.Optional[t.Mapping[casebase.Concept, float]] = None,
) -> t.Optional[casebase.Concept]:
    # Remove the original adaptation source from the candidates
    filter_expr = {rule.source for rule in rules}
    filter_expr.add(original_concept)

    filtered_concepts = {c for c in concepts if c not in filter_expr}
    filtered_concepts = casebase.filter_concepts(
        filtered_concepts, config.tuning("adaptation", "min_concept_score")
    )

    if filtered_concepts:
        # TODO: Check if this makes sense.
        # if occurences and length_differences:
        #     sorted_concepts = sorted(
        #         filtered_concepts,
        #         key=lambda c: 0.5 * c.score
        #         + 0.25 * (_dist2sim(length_differences[c]) or 0.0)
        #         + 0.25 * occurences[c],
        #         reverse=True,
        #     )
        # else:
        sorted_concepts = sorted(
            filtered_concepts,
            key=lambda c: c.score,
            reverse=True,
        )

        return sorted_concepts[0]

    return None


def _filter_paths(
    current_path: graph.AbstractPath,
    path_extensions: t.Iterable[graph.AbstractPath],
    reference_path: graph.AbstractPath,
) -> t.List[graph.AbstractPath]:
    selector = config.tuning("bfs", "selector")
    candidate_values = {}

    start_index = len(current_path.relationships)
    end_index = start_index + 1

    val_reference = _aggregate_features(
        spacy.vector(reference_path.nodes[start_index].processed_name),
        spacy.vector(reference_path.nodes[end_index].processed_name),
        selector,
    )

    for candidate in path_extensions:
        val_adapted = _aggregate_features(
            spacy.vector(current_path.end_node.processed_name),
            spacy.vector(candidate.end_node.processed_name),
            selector,
        )
        candidate_values[candidate] = _compare_features(
            val_reference, val_adapted, selector
        )

    sorted_candidate_tuples = sorted(
        candidate_values.items(), key=lambda x: x[1], reverse=True
    )
    sorted_candidates = [x[0] for x in sorted_candidate_tuples]

    return sorted_candidates[: config["adaptation"]["bfs_node_limit"]]


def _aggregate_features(feat1: t.Any, feat2: t.Any, selector: str) -> t.Any:
    if selector == "difference":
        return abs(feat1 - feat2)
    elif selector == "similarity":
        return spacy.similarity(feat1, feat2)

    raise ValueError("Parameter 'selector' wrong.")


def _compare_features(feat1: t.Any, feat2: t.Any, selector: str) -> t.Any:
    if selector == "difference":
        return spacy.similarity(feat1, feat2)
    elif selector == "similarity":
        return abs(feat1 - feat2)

    raise ValueError("Parameter 'selector' wrong.")
