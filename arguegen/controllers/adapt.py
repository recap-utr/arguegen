from __future__ import annotations

import itertools
import logging
import re
import statistics
import typing as t
from collections import defaultdict
from dataclasses import dataclass

import nlp_service.similarity
import numpy as np
import numpy.typing as npt

from arguegen.config import (
    AdaptationConfig,
    BfsMethod,
    PruningSelector,
    ScoreConfig,
    SubstitutionMethod,
)
from arguegen.controllers import scorer
from arguegen.controllers.inflect import inflect_concept
from arguegen.model import casebase, wordnet
from arguegen.model.nlp import Nlp

log = logging.getLogger(__name__)


def _apply_variants(
    variants: t.Iterable[casebase.ScoredConcept],
    substitutions: t.Mapping[casebase.ScoredConcept, casebase.ScoredConcept],
    graph: casebase.Graph,
    nlp: Nlp,
) -> None:
    for variant in sorted(variants, key=lambda x: len(x.concept.lemma), reverse=True):
        for form, pos_tags in variant.concept.form2pos.items():
            pattern = re.compile(f"\\b({form})\\b", re.IGNORECASE)

            for mapped_node in variant.concept.atoms:
                node = graph.atom_nodes[mapped_node.id]
                pos2form = substitutions[variant].concept.pos2form

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
                                original_text = node.plain_text
                                adapted_text = (
                                    original_text[:start]
                                    + sub_candidates[0]
                                    + original_text[end:]
                                )

                                node.text = adapted_text
                                graph.text.replace(original_text, adapted_text)


def argument_graph(
    case: casebase.Case,
    adaptations: t.Collection[casebase.Rule[casebase.ScoredConcept]],
    nlp: Nlp,
    config: AdaptationConfig,
) -> t.Tuple[casebase.Graph, list[casebase.Rule[casebase.ScoredConcept]]]:
    substitutions = {rule.source: rule.target for rule in adaptations}
    substitutions.update(
        {
            casebase.ScoredConcept(rule.source, 1): casebase.ScoredConcept(
                rule.target, 1
            )
            for rule in case.rules
        }
    )
    substitution_method = config.substitution_method

    sources = list(substitutions.keys())

    if substitution_method == SubstitutionMethod.TARGET_SCORE:
        sources.sort(key=lambda x: substitutions[x].score, reverse=True)
    elif substitution_method == SubstitutionMethod.SOURCE_SCORE:
        sources.sort(key=lambda x: x.score, reverse=True)
    elif substitution_method == SubstitutionMethod.AGGREGATE_SCORE:
        sources.sort(key=lambda x: x.score + substitutions[x].score, reverse=True)

    applied_adaptations: list[casebase.Rule[casebase.ScoredConcept]] = []
    current_adapted_graph = case.case_graph
    current_similarity = nlp.similarity(
        current_adapted_graph.text, case.query_graph.text
    )

    while sources:
        if substitution_method == SubstitutionMethod.QUERY_SIMILARITY:
            adapted_graphs = {}

            for source in sources:
                _adapted_graph = current_adapted_graph.copy()
                variants = frozenset(
                    x for x in sources if source.concept.lemma in x.concept.lemma
                )
                _apply_variants(variants, substitutions, _adapted_graph, nlp)

                adapted_graphs[variants] = (
                    _adapted_graph,
                    nlp.similarity(_adapted_graph.text, case.query_graph.text),
                )

            applied_variants, (new_adapted_graph, new_similarity) = max(
                adapted_graphs.items(), key=lambda x: x[1][1]
            )

        else:
            source = sources[0]
            new_adapted_graph = current_adapted_graph.copy()
            applied_variants = frozenset(
                x for x in sources if source.concept.lemma in x.concept.lemma
            )
            _apply_variants(applied_variants, substitutions, new_adapted_graph, nlp)

            new_similarity = nlp.similarity(
                case.query_graph.text, new_adapted_graph.text
            )

        if new_similarity < current_similarity:
            return current_adapted_graph, applied_adaptations

        current_similarity = new_similarity
        current_adapted_graph = new_adapted_graph

        for variant in applied_variants:
            applied_adaptations.append(casebase.Rule(variant, substitutions[variant]))
            sources.remove(variant)

    return current_adapted_graph, applied_adaptations


def concepts(
    sources: t.Iterable[casebase.ScoredConcept],
    case: casebase.Case,
    nlp: Nlp,
    config: AdaptationConfig,
    score_config: ScoreConfig,
) -> t.Tuple[
    list[casebase.Rule[casebase.ScoredConcept]],
    t.Dict[casebase.ScoredConcept, t.Set[casebase.ScoredConcept]],
]:
    all_candidates = {}
    rules: list[casebase.Rule[casebase.ScoredConcept]] = []

    for source in sources:
        candidate_scores: defaultdict[casebase.Concept, list[float]] = defaultdict(list)
        related_concepts = {source.concept: config.related_concept_weight.original}

        for rule in case.rules:
            related_concepts.update(
                {
                    rule.target: config.related_concept_weight.target / len(case.rules),
                    rule.source: config.related_concept_weight.source / len(case.rules),
                }
            )

        for node in source.concept.synsets:
            hypernym_distances = node.hypernym_distances(
                nlp,
                [atom.plain_text for atom in source.concept.atoms],
                config.node_similarity_threshold,
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
                        source.concept.atoms,
                        nodes,
                    )

                    score = scorer.Scorer(
                        candidate,
                        case.case_graph,
                        case.query_graph,
                        tuple(related_concepts.items()),
                        score_config,
                        nlp,
                        hypernym_level=hyp_distance,
                    ).compute()
                    candidate_scores[candidate].append(score)

        # TODO: Here may be an error!
        max_score = max(max(scores) for scores in candidate_scores.values())

        adaptation_candidates = {
            casebase.ScoredConcept(candidate, max_score)
            for candidate, scores in candidate_scores.items()
            if max(scores) == max_score
        }
        all_candidates[source] = adaptation_candidates

        filtered_adaptations = _filter_concepts(
            adaptation_candidates,
            source,
            [casebase.Rule(rule.source.concept, rule.target.concept) for rule in rules],
            config,
        )
        adapted_lemma = _filter_lemmas(
            filtered_adaptations,
            source,
            [casebase.Rule(rule.source.concept, rule.target.concept) for rule in rules],
            config=config,
            nlp=nlp,
        )

        if adapted_lemma:
            rules.append(casebase.Rule(source, adapted_lemma))
            log.debug(f"Adapt ({source})->({adapted_lemma}).")

        else:
            log.debug(f"No adaptation for ({source}).")

    return rules, all_candidates


def paths(
    reference_paths: t.Mapping[casebase.ScoredConcept, t.Sequence[wordnet.Path]],
    case: casebase.Case,
    nlp: Nlp,
    config: AdaptationConfig,
    score_config: ScoreConfig,
) -> t.Tuple[
    list[casebase.Rule[casebase.ScoredConcept]],
    t.Dict[casebase.ScoredConcept, t.List[wordnet.Path]],
    t.Dict[casebase.ScoredConcept, t.Set[casebase.ScoredConcept]],
]:
    rules = []
    adapted_paths = {}
    all_candidates = {}

    for source, all_shortest_paths in reference_paths.items():
        start_nodes = (
            itertools.chain.from_iterable(rule.target.synsets for rule in case.rules)
            if config.bfs_method == BfsMethod.WITHIN
            else source.concept.synsets
        )

        adaptation_results = [
            _bfs_adaptation(shortest_path, source, start_nodes, nlp, config)
            for shortest_path in all_shortest_paths
        ]

        adaptation_results = list(itertools.chain.from_iterable(adaptation_results))
        adapted_paths[source] = adaptation_results
        candidate_scores = defaultdict(list)

        for result in adaptation_results:
            hyp_distance = len(result)
            pos = result.end_node.pos
            end_nodes = frozenset([result.end_node])
            related_concepts = {source.concept: config.related_concept_weight.original}

            for lemma in result.end_node.lemmas:
                _, form2pos, pos2form = inflect_concept(
                    lemma, casebase.pos2spacy(pos), lemmatize=False
                )

                for rule in case.rules:
                    related_concepts.update(
                        {
                            rule.target: config.related_concept_weight.target / len(
                                case.rules
                            ),
                            rule.source: config.related_concept_weight.source / len(
                                case.rules
                            ),
                        }
                    )

                candidate = casebase.Concept(
                    lemma,
                    form2pos,
                    pos2form,
                    pos,
                    source.concept.atoms,
                    end_nodes,
                )
                score = scorer.Scorer(
                    candidate,
                    case.case_graph,
                    case.query_graph,
                    tuple(related_concepts.items()),
                    score_config,
                    nlp,
                    hypernym_level=hyp_distance,
                ).compute()
                candidate_scores[candidate].append(score)

        # TODO: Here may be an error!
        max_score = max(max(scores) for scores in candidate_scores.values())

        adaptation_candidates = {
            casebase.ScoredConcept(candidate, max_score)
            for candidate, scores in candidate_scores.items()
            if max(scores) == max_score
        }
        all_candidates[source] = adaptation_candidates

        filtered_adaptations = _filter_concepts(
            adaptation_candidates, source, case.rules, config
        )
        adapted_lemma = _filter_lemmas(
            filtered_adaptations, source, case.rules, config, nlp
        )

        if adapted_lemma:
            rules.append(casebase.Rule(source, adapted_lemma))
            log.debug(f"Adapt ({source})->({adapted_lemma}).")

        else:
            log.debug(f"No adaptation for ({source}).")

    return rules, adapted_paths, all_candidates


def _bfs_adaptation(
    shortest_path: wordnet.Path,
    source: casebase.ScoredConcept,
    start_nodes: t.Iterable[wordnet.Synset],
    nlp: Nlp,
    config: AdaptationConfig,
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
                    nlp,
                    current_path.end_node,
                    [atom.plain_text for atom in source.concept.atoms],
                    config.node_similarity_threshold,
                )

                # Here, paths that are shorter than the reference path are discarded.
                if path_extensions:
                    path_extensions = _filter_paths(
                        path_extensions, current_path, shortest_path, nlp, config
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
    concepts: t.AbstractSet[casebase.ScoredConcept],
    source: casebase.ScoredConcept,
    rules: t.Collection[casebase.Rule[casebase.Concept]],
    config: AdaptationConfig,
    limit: t.Optional[int] = None,
) -> t.List[casebase.ScoredConcept]:
    # Remove the original adaptation source from the candidates
    filter_expr = {rule.source for rule in rules}
    filter_expr.add(source.concept)

    filtered_concepts = {c for c in concepts if c.concept not in filter_expr}
    filtered_concepts = scorer.filter_concepts(
        filtered_concepts,
        config.concept_score_threshold,
        None,
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
    pos: casebase.Pos.ValueType
    nodes: t.FrozenSet[wordnet.Synset]
    concepts: t.FrozenSet[casebase.Concept]


def _filter_lemmas(
    targets: t.Sequence[casebase.ScoredConcept],
    source: casebase.ScoredConcept,
    rules: t.Iterable[casebase.Rule[casebase.Concept]],
    config: AdaptationConfig,
    nlp: Nlp,
) -> t.Optional[casebase.ScoredConcept]:
    if not targets:
        return None

    max_lemmas = config.lemma_limit
    lemma_nodes = defaultdict(set)
    lemma_concepts = defaultdict(set)

    for target in targets[:max_lemmas]:
        for node in target.concept.synsets:
            for lemma in node.lemmas:
                lemma_nodes[(lemma, target.concept.pos)].add(node)
                lemma_concepts[(lemma, target.concept.pos)].add(target)

    assert lemma_nodes.keys() == lemma_concepts.keys()

    lemmas = [
        Lemma(name, pos, frozenset(nodes), frozenset(concepts))
        for ((name, pos), nodes), concepts in zip(
            lemma_nodes.items(), lemma_concepts.values()
        )
    ]

    best_lemmas = _prune(
        lemmas,
        source.concept,
        [(rule.source, rule.target) for rule in rules],
        limit=1,
        nlp=nlp,
        selector=config.pruning_selector,
    )

    if len(best_lemmas) == 0:
        return None

    best_lemma = best_lemmas[0]

    _lemma, _form2pos, _pos2form = inflect_concept(
        best_lemma.lemma, casebase.pos2spacy(best_lemma.pos), lemmatize=False
    )

    return casebase.ScoredConcept(
        casebase.Concept(
            lemma=_lemma,
            form2pos=_form2pos,
            pos2form=_pos2form,
            _pos=best_lemma.pos,
            atoms=source.concept.atoms,
            synsets=best_lemma.nodes,
        ),
        0,
    )


def _filter_paths(
    path_extensions: t.Iterable[wordnet.Path],
    current_path: wordnet.Path,
    reference_path: wordnet.Path,
    nlp: Nlp,
    config: AdaptationConfig,
) -> t.List[wordnet.Path]:
    start_index = len(current_path.relationships)
    end_index = start_index + 1

    return _prune(
        path_extensions,
        current_path,
        [(reference_path.nodes[start_index], reference_path.nodes[end_index])],
        limit=config.pruning_bfs_limit,
        nlp=nlp,
        selector=config.pruning_selector,
    )


_AdaptedItem = t.TypeVar("_AdaptedItem", Lemma, wordnet.Path)
_ReferenceItem = t.Union[wordnet.Synset, casebase.Concept]
_RetrievedItem = t.Union[wordnet.Synset, casebase.Concept, wordnet.Path]


def _prune(
    adapted_items: t.Iterable[_AdaptedItem],
    retrieved_item: _RetrievedItem,
    reference_items: t.Iterable[t.Tuple[_ReferenceItem, _ReferenceItem]],
    selector: PruningSelector,
    nlp: Nlp,
    limit: t.Optional[int] = None,
) -> t.List[_AdaptedItem]:
    candidate_values = defaultdict(list)

    for reference_source, reference_target in reference_items:
        val_reference = _aggregate_features(
            reference_source.lemma, reference_target.lemma, selector, nlp
        )

        for adapted_item in adapted_items:
            val_adapted = _aggregate_features(
                retrieved_item.lemma, adapted_item.lemma, selector, nlp
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
    feat1: str, feat2: str, selector: PruningSelector, nlp: Nlp
) -> t.Union[float, npt.NDArray[np.float_]]:
    if selector == PruningSelector.SIMILARITY:
        return nlp.similarity(feat1, feat2)
    elif selector == PruningSelector.DIFFERENCE:
        return nlp.vector(feat1) - nlp.vector(feat2)

    raise ValueError("Parameter 'selector' wrong.")


def _compare_features(
    feat1: t.Union[float, npt.NDArray[np.float_]],
    feat2: t.Union[float, npt.NDArray[np.float_]],
    selector: PruningSelector,
) -> float:
    if selector == PruningSelector.SIMILARITY:
        assert isinstance(feat1, float) and isinstance(feat2, float)
        return 1 - abs(feat1 - feat2)
    elif selector == PruningSelector.DIFFERENCE:
        assert isinstance(feat1, np.ndarray) and isinstance(feat2, np.ndarray)
        return nlp_service.similarity.cosine(feat1, feat2)

    raise ValueError("Parameter 'selector' wrong.")
