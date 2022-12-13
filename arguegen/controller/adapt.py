from __future__ import annotations

import itertools
import logging
import re
import statistics
import typing as t
from collections import defaultdict
from dataclasses import dataclass

import arguebuf as ag
import nlp_service.similarity
import numpy as np
import numpy.typing as npt
from scipy.spatial import distance

from arguegen.config import config, tuning
from arguegen.controller.inflect import inflect_concept
from arguegen.model import casebase, evaluation, nlp, wordnet

log = logging.getLogger(__name__)


def _graph_similarity(user_query: casebase.UserQuery, graph: ag.Graph) -> float:
    graph_text = " ".join(atom.plain_text for atom in graph.atom_nodes.values())

    return nlp.similarity(user_query.text, graph_text)


def _apply_variants(
    variants: t.Iterable[casebase.Concept],
    substitutions: t.Mapping[casebase.Concept, casebase.Concept],
    adapted_graph: ag.Graph,
) -> None:
    for variant in sorted(variants, key=lambda x: len(x.lemma), reverse=True):
        for form, pos_tags in variant.form2pos.items():
            pattern = re.compile(f"\\b({form})\\b", re.IGNORECASE)

            for mapped_node in variant.atoms:
                node = adapted_graph.atom_nodes[mapped_node.id]
                pos2form = substitutions[variant].pos2form

                for pos_tag in pos_tags:
                    if pos_tag in pos2form:
                        sub_candidates = pos2form[pos_tag]

                        for match in re.finditer(pattern, node.text):
                            node_doc = nlp.parse_doc(
                                node.plain_text,
                                attributes=["POS", "TAG"],
                            )
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


def argument_graph(
    user_query: casebase.UserQuery,
    original_graph: ag.Graph,
    rules: t.Collection[casebase.Rule],
    adapted_concepts: t.Mapping[casebase.Concept, casebase.Concept],
) -> t.Tuple[ag.Graph, t.Mapping[casebase.Concept, casebase.Concept]]:
    substitutions = {**adapted_concepts}
    substitutions.update({rule.source: rule.target for rule in rules})
    substitution_method = tuning(config, "adaptation", "substitution_method")

    sources = list(substitutions.keys())

    if substitution_method == "target_score":
        sources.sort(key=lambda x: substitutions[x].score, reverse=True)
    elif substitution_method == "source_score":
        sources.sort(key=lambda x: x.score, reverse=True)
    elif substitution_method == "score":
        sources.sort(key=lambda x: x.score + substitutions[x].score, reverse=True)

    applied_adaptations = {}
    current_similarity = _graph_similarity(user_query, original_graph)
    current_adapted_graph = original_graph

    while sources:
        if substitution_method == "query_sim":
            adapted_graphs = {}

            for source in sources:
                _adapted_graph = current_adapted_graph.copy()
                variants = frozenset(x for x in sources if source.lemma in x.lemma)
                _apply_variants(variants, substitutions, _adapted_graph)

                adapted_graphs[variants] = (
                    _adapted_graph,
                    _graph_similarity(user_query, _adapted_graph),
                )

            applied_variants, (new_adapted_graph, new_similarity) = max(
                adapted_graphs.items(), key=lambda x: x[1][1]
            )

        elif substitution_method in ["source_score", "target_score", "score"]:
            source = sources[0]
            new_adapted_graph = current_adapted_graph.copy()
            applied_variants = frozenset(x for x in sources if source.lemma in x.lemma)
            _apply_variants(applied_variants, substitutions, new_adapted_graph)

            new_similarity = _graph_similarity(user_query, new_adapted_graph)

        else:
            raise ValueError(f"Setting {substitution_method=} is not valid.")

        if new_similarity < current_similarity:
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
    related_concept_weight = tuning(config, "adaptation_weight")

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

        for node in original_concept.synsets:
            hypernym_distances = node.hypernym_distances(
                [atom.plain_text for atom in original_concept.atoms],
                tuning(config, "threshold", "node_similarity", "adaptation"),
            )

            for hypernym, hyp_distance in hypernym_distances.items():
                nodes = frozenset([hypernym])

                for lemma in hypernym.lemmas:
                    _, form2pos, pos2form = inflect_concept(
                        lemma, casebase.pos2spacy(hypernym.pos), lemmatize=False
                    )

                    candidate = casebase.Concept(
                        lemma,
                        form2pos,
                        pos2form,
                        hypernym.pos,
                        original_concept.atoms,
                        nodes,
                        related_concepts,
                        user_query,
                        evaluation.concept_metrics(
                            "adaptation",
                            related_concepts,
                            user_query,
                            original_concept.atoms,
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
    reference_paths: t.Mapping[casebase.Concept, t.Sequence[wordnet.Path]],
    rules: t.Collection[casebase.Rule],
    user_query: casebase.UserQuery,
) -> t.Tuple[
    t.Dict[casebase.Concept, casebase.Concept],
    t.Dict[casebase.Concept, t.List[wordnet.Path]],
    t.Dict[casebase.Concept, t.Set[casebase.Concept]],
]:
    bfs_method = "between"
    related_concept_weight = tuning(config, "adaptation_weight")
    adapted_concepts = {}
    adapted_paths = {}
    all_candidates = {}

    for original_concept, all_shortest_paths in reference_paths.items():
        start_nodes = (
            itertools.chain.from_iterable(rule.target.synsets for rule in rules)
            if bfs_method == "within"
            else original_concept.synsets
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
            pos = result.end_node.pos
            end_nodes = frozenset([result.end_node])
            related_concepts = {
                original_concept: related_concept_weight.get("original_concept", 0.0)
            }

            for lemma in result.end_node.lemmas:
                _, form2pos, pos2form = inflect_concept(
                    lemma, casebase.pos2spacy(pos), lemmatize=False
                )

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
                    original_concept.atoms,
                    end_nodes,
                    related_concepts,
                    user_query,
                    evaluation.concept_metrics(
                        "adaptation",
                        related_concepts,
                        user_query,
                        original_concept.atoms,
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
    shortest_path: wordnet.Path,
    concept: casebase.Concept,
    start_nodes: t.Iterable[wordnet.Node],
) -> t.Set[wordnet.Path]:
    adapted_paths = set()

    for start_node in start_nodes:
        current_paths = set()

        current_paths.add(
            wordnet.Path.from_node(start_node)
        )  # Start with only one node.

        for _ in shortest_path.relationships:
            next_paths = set()

            for current_path in current_paths:
                path_extensions = wordnet.direct_hypernyms(
                    current_path.end_node,
                    [atom.plain_text for atom in concept.atoms],
                    tuning(config, "threshold", "node_similarity", "adaptation"),
                )

                # Here, paths that are shorter than the reference path are discarded.
                if path_extensions:
                    path_extensions = _filter_paths(
                        path_extensions, current_path, shortest_path
                    )

                    for path_extension in path_extensions:
                        next_paths.add(wordnet.Path.merge(current_path, path_extension))

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
        tuning(config, "threshold", "concept_score", "adaptation"),
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
    lemma: str
    pos: casebase.POS
    nodes: t.FrozenSet[wordnet.Node]
    concepts: t.FrozenSet[casebase.Concept]


def _filter_lemmas(
    adapted_concepts: t.Sequence[casebase.Concept],
    retrieved_concept: casebase.Concept,
    rules: t.Iterable[casebase.Rule],
) -> t.Optional[casebase.Concept]:
    if not adapted_concepts:
        return None

    max_lemmas = tuning(config, "adaptation", "lemma_limit")
    lemma_nodes = defaultdict(set)
    lemma_concepts = defaultdict(set)

    for adapted_concept in adapted_concepts[:max_lemmas]:
        for node in adapted_concept.synsets:
            for lemma in node.lemmas:
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
        best_lemma.lemma, casebase.pos2spacy(best_lemma.pos), lemmatize=False
    )

    return casebase.Concept(
        lemma=_lemma,
        form2pos=_form2pos,
        pos2form=_pos2form,
        pos=best_lemma.pos,
        atoms=retrieved_concept.atoms,
        synsets=best_lemma.nodes,
        related_concepts={},
        user_query=retrieved_concept.user_query,
        metrics=casebase.empty_metrics(),
    )


def _filter_paths(
    path_extensions: t.Iterable[wordnet.Path],
    current_path: wordnet.Path,
    reference_path: wordnet.Path,
) -> t.List[wordnet.Path]:
    start_index = len(current_path.relationships)
    end_index = start_index + 1

    return _prune(
        path_extensions,
        current_path,
        [(reference_path.nodes[start_index], reference_path.nodes[end_index])],
        limit=tuning(config, "adaptation", "pruning_bfs_limit"),
    )


def _prune(
    adapted_items: t.Iterable[t.Any],
    retrieved_item: t.Any,
    reference_items: t.Iterable[t.Tuple[t.Any, t.Any]],
    limit: t.Optional[int] = None,
) -> t.List[t.Any]:
    candidate_values = defaultdict(list)
    selector = tuning(config, "adaptation", "pruning_selector")

    for item in reference_items:
        val_reference = _aggregate_features(
            item[0].lemma,
            item[1].lemma,
            selector,
        )

        for adapted_item in adapted_items:
            val_adapted = _aggregate_features(
                retrieved_item.lemma,
                adapted_item.lemma,
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
) -> t.Union[float, npt.NDArray[np.float_]]:
    if selector == "similarity":
        return nlp.similarity(feat1, feat2)
    elif selector == "difference":
        return nlp.vector(feat1) - nlp.vector(feat2)

    raise ValueError("Parameter 'selector' wrong.")


def _compare_features(
    feat1: t.Union[float, npt.NDArray[np.float_]],
    feat2: t.Union[float, npt.NDArray[np.float_]],
    selector: str,
) -> float:
    if selector == "similarity":
        assert isinstance(feat1, float) and isinstance(feat2, float)
        return 1 - abs(feat1 - feat2)
    elif selector == "difference":
        assert isinstance(feat1, np.ndarray) and isinstance(feat2, np.ndarray)
        return nlp_service.similarity.cosine(feat1, feat2)

    raise ValueError("Parameter 'selector' wrong.")
