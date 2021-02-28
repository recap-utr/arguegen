from __future__ import annotations

import itertools
import logging
import re
import statistics
import typing as t
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import recap_argument_graph as ag
from recap_argument_graph_adaptation.controller.inflect import inflect_concept
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
        adaptation_map = defaultdict(list)
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
                config.tuning("threshold", "nodes_similarity", "adaptation"),
            )

            for hypernym, hyp_distance in hypernym_distances.items():
                name = hypernym.processed_name
                pos = hypernym.pos
                vector = spacy.vector(name)
                nodes = frozenset([hypernym])

                candidate = casebase.Concept(
                    name,
                    vector,
                    frozenset([name]),
                    query.pos(pos),
                    original_concept.inodes,
                    nodes,
                    related_concepts,
                    user_query,
                    query.concept_metrics(
                        related_concepts,
                        user_query,
                        original_concept.inodes,
                        nodes,
                        vector,
                        hypernym_level=hyp_distance,
                    ),
                )

                adaptation_map[str(candidate)].append(candidate)

        adaptation_candidates = {
            max(candidates, key=lambda x: x.score)
            for candidates in adaptation_map.values()
        }
        all_candidates[original_concept] = adaptation_candidates

        filtered_adaptations = _filter_concepts(
            adaptation_candidates, original_concept, rules
        )
        adapted_lemma = _filter_lemmas(
            filtered_adaptations[: config.tuning("adaptation", "concept_candidates_limit")],
            original_concept,
            rules,
        )

        if adapted_lemma:
            adapted_concepts[original_concept] = adapted_lemma
            log.debug(f"Adapt ({original_concept})->({adapted_lemma}).")

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
    bfs_method = "between"
    related_concept_weight = config.tuning("weight")
    adapted_concepts = {}
    adapted_paths = {}
    all_candidates = {}

    for original_concept, all_shortest_paths in reference_paths.items():
        start_nodes = (
            itertools.chain.from_iterable(rule.target.nodes for rule in rules)
            if bfs_method == "within"
            else original_concept.nodes
        )

        adaptation_results = [
            _bfs_adaptation(shortest_path, original_concept, start_nodes)
            for shortest_path in all_shortest_paths
        ]

        adaptation_results = list(itertools.chain.from_iterable(adaptation_results))
        adapted_paths[original_concept] = adaptation_results
        adaptation_map = defaultdict(list)

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
                frozenset([name]),
                pos,
                original_concept.inodes,
                end_nodes,
                related_concepts,
                user_query,
                query.concept_metrics(
                    related_concepts,
                    user_query,
                    original_concept.inodes,
                    end_nodes,
                    vector,
                    hypernym_level=hyp_distance,
                ),
            )

            adaptation_map[str(candidate)].append(candidate)

        adaptation_candidates = {
            max(candidates, key=lambda x: x.score)
            for candidates in adaptation_map.values()
        }
        all_candidates[original_concept] = adaptation_candidates

        filtered_adaptations = _filter_concepts(
            adaptation_candidates, original_concept, rules
        )
        adapted_lemma = _filter_lemmas(
            filtered_adaptations[: config.tuning("adaptation", "concept_candidates_limit")],
            original_concept,
            rules,
        )

        if adapted_lemma:
            adapted_concepts[original_concept] = adapted_lemma
            log.debug(f"Adapt ({original_concept})->({adapted_lemma}).")

        else:
            log.debug(f"No adaptation for ({original_concept}).")

    return adapted_concepts, adapted_paths, all_candidates


def _bfs_adaptation(
    shortest_path: graph.AbstractPath,
    concept: casebase.Concept,
    start_nodes: t.Iterable[graph.AbstractNode],
) -> t.Set[graph.AbstractPath]:
    adapted_paths = set()

    for start_node in start_nodes:
        current_paths = set()

        current_paths.add(
            graph.AbstractPath.from_node(start_node)
        )  # Start with only one node.

        for _ in shortest_path.relationships:
            next_paths = set()

            for current_path in current_paths:
                path_extensions = query.direct_hypernyms(
                    current_path.end_node,
                    concept.inode_vectors,
                    config.tuning("threshold", "nodes_similarity", "adaptation"),
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

                # Shorter paths are removed.
                # In case you want these, uncomment the following lines.
                # else:
                #     adapted_paths.add(current_path)

            current_paths = next_paths

        adapted_paths.update(current_paths)

    return adapted_paths


def _filter_concepts(
    concepts: t.Set[casebase.Concept],
    original_concept: casebase.Concept,
    rules: t.Collection[casebase.Rule],
    limit: t.Optional[int] = None,
) -> t.List[casebase.Concept]:
    # Remove the original adaptation source from the candidates
    filter_expr = {rule.source for rule in rules}
    filter_expr.add(original_concept)

    filtered_concepts = {c for c in concepts if c not in filter_expr}
    filtered_concepts = casebase.filter_concepts(
        filtered_concepts,
        config.tuning("threshold", "concept_score", "adaptation"),
        topn=None,
    )

    if filtered_concepts:
        sorted_concepts = sorted(
            filtered_concepts,
            key=lambda c: c.score,
            reverse=True,
        )

        if limit:
            return sorted_concepts[:limit]

        return sorted_concepts

    return []


@dataclass(frozen=True)
class Lemma:
    name: str
    pos: casebase.POS
    nodes: t.FrozenSet[graph.AbstractNode]
    concepts: t.FrozenSet[casebase.Concept]
    vector: spacy.Vector = field(repr=False, compare=False)


def _filter_lemmas(
    adapted_concepts: t.Iterable[casebase.Concept],
    retrieved_concept: casebase.Concept,
    rules: t.Iterable[casebase.Rule],
) -> t.Optional[casebase.Concept]:
    if not adapted_concepts:
        return None

    lemma_nodes = defaultdict(set)
    lemma_concepts = defaultdict(set)

    for adapted_concept in adapted_concepts:
        for node in adapted_concept.nodes:
            for lemma in node.processed_lemmas:
                lemma_nodes[(lemma, adapted_concept.pos)].add(node)
                lemma_concepts[(lemma, adapted_concept.pos)].add(adapted_concept)

    assert lemma_nodes.keys() == lemma_concepts.keys()

    lemma_vectors = spacy.vectors(
        [lemma_tuple[0] for lemma_tuple in lemma_nodes.keys()]
    )
    lemmas = [
        Lemma(name, pos, frozenset(nodes), frozenset(concepts), vector)
        for ((name, pos), nodes), concepts, vector in zip(
            lemma_nodes.items(), lemma_concepts.values(), lemma_vectors
        )
    ]

    best_lemma: Lemma = _prune(
        lemmas,
        retrieved_concept,
        [(rule.source, rule.target) for rule in rules],
        selector=config.tuning("adaptation", "pruning_selector"),
        limit=1,
    )[0]

    return casebase.Concept(
        name=best_lemma.name,
        forms=frozenset([best_lemma.name]),
        pos=best_lemma.pos,
        inodes=retrieved_concept.inodes,
        nodes=best_lemma.nodes,
        related_concepts={},
        user_query=retrieved_concept.user_query,
        vector=best_lemma.vector,
        metrics=casebase.empty_metrics(),
    )


def _filter_paths(
    current_path: graph.AbstractPath,
    path_extensions: t.Iterable[graph.AbstractPath],
    reference_path: graph.AbstractPath,
) -> t.List[graph.AbstractPath]:
    start_index = len(current_path.relationships)
    end_index = start_index + 1

    return _prune(
        path_extensions,
        current_path,
        [(reference_path.nodes[start_index], reference_path.nodes[end_index])],
        selector=config.tuning("adaptation", "pruning_selector"),
        limit=config.tuning("adaptation", "pruning_bfs_limit"),
    )


def _prune(
    adapted_items: t.Iterable[t.Any],
    retrieved_item: t.Any,
    reference_items: t.Iterable[t.Tuple[t.Any, t.Any]],
    selector: str,
    limit: t.Optional[int] = None,
) -> t.List[t.Any]:
    candidate_values = defaultdict(list)

    for item in reference_items:
        val_reference = _aggregate_features(
            item[0].vector,
            item[1].vector,
            selector,
        )

        for adapted_item in adapted_items:
            val_adapted = _aggregate_features(
                retrieved_item.vector,
                adapted_item.vector,
                selector,
            )
            candidate_values[adapted_item].append(
                _compare_features(val_reference, val_adapted, selector)
            )

    sorted_candidate_tuples = sorted(
        candidate_values.items(), key=lambda x: statistics.mean(x[1]), reverse=True
    )
    sorted_candidates = [x[0] for x in sorted_candidate_tuples]

    if limit:
        return sorted_candidates[:limit]

    return sorted_candidates


def _aggregate_features(feat1: t.Any, feat2: t.Any, selector: str) -> t.Any:
    if selector == "difference":
        return 1 - abs(feat1 - feat2)
    elif selector == "similarity":
        return spacy.similarity(feat1, feat2)

    raise ValueError("Parameter 'selector' wrong.")


def _compare_features(feat1: t.Any, feat2: t.Any, selector: str) -> t.Any:
    if selector == "difference":
        return spacy.similarity(feat1, feat2)
    elif selector == "similarity":
        return 1 - abs(feat1 - feat2)

    raise ValueError("Parameter 'selector' wrong.")
