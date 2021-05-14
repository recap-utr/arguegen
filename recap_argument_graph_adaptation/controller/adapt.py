from __future__ import annotations

import itertools
import logging
import re
import statistics
import typing as t
from collections import defaultdict
from dataclasses import dataclass, field

import nltk
import numpy as np
import recap_argument_graph as ag
import spacy
from recap_argument_graph_adaptation.controller.inflect import inflect_concept
from recap_argument_graph_adaptation.model import casebase, graph, nlp, query
from recap_argument_graph_adaptation.model.config import Config
from scipy.spatial import distance

config = Config.instance()

log = logging.getLogger(__name__)


def _graph_similarity(user_query: casebase.UserQuery, graph: ag.Graph) -> float:
    graph_text = " ".join(inode.plain_text for inode in graph.inodes)

    return nlp.similarity(user_query.text, graph_text)


def argument_graph(
    user_query: casebase.UserQuery,
    original_graph: ag.Graph,
    rules: t.Collection[casebase.Rule],
    adapted_concepts: t.Mapping[casebase.Concept, casebase.Concept],
) -> t.Tuple[ag.Graph, t.Mapping[casebase.Concept, casebase.Concept]]:
    original_similarity = _graph_similarity(user_query, original_graph)
    spacy_nlp = spacy.load("en_core_web_sm")

    substitutions = {**adapted_concepts}
    substitutions.update({rule.source: rule.target for rule in rules})
    # sources = sorted(substitutions.keys(), key=lambda x: x.score, reverse=True)
    sources = set(substitutions.keys())
    applied_adaptations = {}
    current_similarity = 0.0
    current_adapted_graph = original_graph

    while sources:
        adapted_graphs = {}

        for source in sources:
            _adapted_graph = current_adapted_graph.copy()
            variants = frozenset(x for x in sources if source.name in x.name)

            for variant in sorted(variants, key=lambda x: len(x.name), reverse=True):
                for form, pos_tags in variant.form2pos.items():
                    pattern = re.compile(f"\\b({form})\\b", re.IGNORECASE)

                    for mapped_node in variant.inodes:
                        node = _adapted_graph.inode_mappings[mapped_node.key]
                        pos2form = substitutions[variant].pos2form

                        for pos_tag in pos_tags:
                            if pos_tag in pos2form:
                                sub_candidates = pos2form[pos_tag]

                                for match in re.finditer(pattern, node.text):
                                    node_doc = spacy_nlp(node.plain_text)
                                    # node_doc = nlp.parse_doc(
                                    #     node.plain_text,
                                    #     attributes=["POS", "TAG"],
                                    # )
                                    start, end = match.span()
                                    span = node_doc.char_span(start, end)

                                    if span is not None and any(
                                        t.tag_ == pos_tag for t in span
                                    ):
                                        node.text = (
                                            node.plain_text[:start]
                                            + sub_candidates[0]
                                            + node.plain_text[end:]
                                        )

            adapted_graphs[variants] = (
                _adapted_graph,
                _graph_similarity(user_query, _adapted_graph),
            )

        applied_variants, (new_adapted_graph, new_similarity) = max(
            adapted_graphs.items(), key=lambda x: x[1][1]
        )

        if new_similarity < current_similarity or new_similarity < original_similarity:
            return current_adapted_graph, applied_adaptations

        current_similarity = new_similarity
        current_adapted_graph = new_adapted_graph

        for variant in applied_variants:
            applied_adaptations[variant] = substitutions[variant]
            sources.remove(variant)

    return current_adapted_graph, applied_adaptations


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
    related_concept_weight = config.tuning("adaptation_weight")

    for original_concept in concepts:
        adaptation_map = defaultdict(list)
        related_concepts = {
            original_concept: related_concept_weight["original_concept"]
        }

        for rule in rules:
            related_concepts.update(
                {
                    rule.target: related_concept_weight.get("rule_target", 0.0)
                    / len(rules),
                    rule.source: related_concept_weight.get("rule_source", 0.0)
                    / len(rules),
                }
            )

        for node in original_concept.nodes:
            hypernym_distances = node.hypernym_distances(
                [inode.plain_text for inode in original_concept.inodes],
                config.tuning("threshold", "node_similarity", "adaptation"),
            )

            for hypernym, hyp_distance in hypernym_distances.items():
                name = hypernym.processed_name
                pos = query.pos(hypernym.pos)
                nodes = frozenset([hypernym])
                lemma, form2pos, pos2form = inflect_concept(
                    name, casebase.pos2spacy(pos)
                )

                candidate = casebase.Concept(
                    lemma,
                    form2pos,
                    pos2form,
                    pos,
                    original_concept.inodes,
                    nodes,
                    related_concepts,
                    user_query,
                    query.concept_metrics(
                        "adaptation",
                        related_concepts,
                        user_query,
                        original_concept.inodes,
                        nodes,
                        lemma,
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
        adapted_lemma = _filter_lemmas(filtered_adaptations, original_concept, rules)

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
    related_concept_weight = config.tuning("adaptation_weight")
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
            end_nodes = frozenset([result.end_node])
            pos = query.pos(result.end_node.pos)
            related_concepts = {
                original_concept: related_concept_weight.get("original_concept", 0.0)
            }
            lemma, form2pos, pos2form = inflect_concept(name, casebase.pos2spacy(pos))

            for rule in rules:
                related_concepts.update(
                    {
                        rule.target: related_concept_weight.get("rule_target", 0.0)
                        / len(rules),
                        rule.source: related_concept_weight.get("rule_source", 0.0)
                        / len(rules),
                    }
                )

            candidate = casebase.Concept(
                lemma,
                form2pos,
                pos2form,
                pos,
                original_concept.inodes,
                end_nodes,
                related_concepts,
                user_query,
                query.concept_metrics(
                    "adaptation",
                    related_concepts,
                    user_query,
                    original_concept.inodes,
                    end_nodes,
                    lemma,
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
        adapted_lemma = _filter_lemmas(filtered_adaptations, original_concept, rules)

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
                    [inode.plain_text for inode in concept.inodes],
                    config.tuning("threshold", "node_similarity", "adaptation"),
                )

                # Here, paths that are shorter than the reference path are discarded.
                if path_extensions:
                    path_extensions = _filter_paths(
                        path_extensions, current_path, shortest_path
                    )

                    for path_extension in path_extensions:
                        next_paths.add(
                            graph.AbstractPath.merge(current_path, path_extension)
                        )

                # In case you want shorter paths, uncomment the following lines.
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


class BreakLoop(Exception):
    pass


@dataclass(frozen=True)
class Lemma:
    name: str
    pos: casebase.POS
    nodes: t.FrozenSet[graph.AbstractNode]
    concepts: t.FrozenSet[casebase.Concept]


def _filter_lemmas(
    adapted_concepts: t.Sequence[casebase.Concept],
    retrieved_concept: casebase.Concept,
    rules: t.Iterable[casebase.Rule],
) -> t.Optional[casebase.Concept]:
    if not adapted_concepts:
        return None

    max_lemmas = config.tuning("adaptation", "lemma_limit")
    lemma_nodes = defaultdict(set)
    lemma_concepts = defaultdict(set)

    for adapted_concept in adapted_concepts[:max_lemmas]:
        for node in adapted_concept.nodes:
            for lemma in node.processed_lemmas:
                lemma_nodes[(lemma, adapted_concept.pos)].add(node)
                lemma_concepts[(lemma, adapted_concept.pos)].add(adapted_concept)

    assert lemma_nodes.keys() == lemma_concepts.keys()

    lemmas = [
        Lemma(name, pos, frozenset(nodes), frozenset(concepts))
        for ((name, pos), nodes), concepts in zip(
            lemma_nodes.items(), lemma_concepts.values()
        )
    ]

    best_lemma: Lemma = _prune(
        lemmas,
        retrieved_concept,
        [(rule.source, rule.target) for rule in rules],
        limit=1,
    )[0]

    _lemma, _form2pos, _pos2form = inflect_concept(
        best_lemma.name, casebase.pos2spacy(best_lemma.pos)
    )

    return casebase.Concept(
        name=_lemma,
        form2pos=_form2pos,
        pos2form=_pos2form,
        pos=best_lemma.pos,
        inodes=retrieved_concept.inodes,
        nodes=best_lemma.nodes,
        related_concepts={},
        user_query=retrieved_concept.user_query,
        metrics=casebase.empty_metrics(),
    )


def _filter_paths(
    path_extensions: t.Iterable[graph.AbstractPath],
    current_path: graph.AbstractPath,
    reference_path: graph.AbstractPath,
) -> t.List[graph.AbstractPath]:
    start_index = len(current_path.relationships)
    end_index = start_index + 1

    return _prune(
        path_extensions,
        current_path,
        [(reference_path.nodes[start_index], reference_path.nodes[end_index])],
        limit=config.tuning("adaptation", "pruning_bfs_limit"),
    )


def _prune(
    adapted_items: t.Iterable[t.Any],
    retrieved_item: t.Any,
    reference_items: t.Iterable[t.Tuple[t.Any, t.Any]],
    limit: t.Optional[int] = None,
) -> t.List[t.Any]:
    candidate_values = defaultdict(list)
    selector = config.tuning("adaptation", "pruning_selector")

    if nlp.use_token_vectors():
        selector = "similarity"

    for item in reference_items:
        val_reference = _aggregate_features(
            item[0].name,
            item[1].name,
            selector,
        )

        for adapted_item in adapted_items:
            val_adapted = _aggregate_features(
                retrieved_item.name,
                adapted_item.name,
                selector,
            )
            candidate_values[adapted_item].append(
                _compare_features(val_reference, val_adapted, selector)
            )

    sorted_candidate_tuples = sorted(
        candidate_values.items(), key=lambda x: statistics.mean(x[1]), reverse=True
    )
    sorted_candidates = [x[0] for x in sorted_candidate_tuples]

    if limit and limit > 0:
        return sorted_candidates[:limit]

    return sorted_candidates


def _aggregate_features(
    feat1: str, feat2: str, selector: str
) -> t.Union[float, np.ndarray]:
    if selector == "difference":
        return nlp.vector(feat1) - nlp.vector(feat2)  # type: ignore
    elif selector == "similarity":
        return nlp.similarity(feat1, feat2)

    raise ValueError("Parameter 'selector' wrong.")


def _compare_features(
    feat1: t.Union[float, np.ndarray], feat2: t.Union[float, np.ndarray], selector: str
) -> float:
    if selector == "similarity":
        return 1 - abs(feat1 - feat2)
    elif selector == "difference":
        return 1 - distance.cosine(feat1, feat2)  # type: ignore

    raise ValueError("Parameter 'selector' wrong.")
