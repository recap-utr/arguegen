import itertools
import logging
import re
import typing as t
from collections import defaultdict

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
                config.tuning("threshold", "synset_similarity", "adaptation"),
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

                adaptation_candidates.add(candidate)

        all_candidates[original_concept] = adaptation_candidates
        adapted_concept = _filter_concepts(
            adaptation_candidates, original_concept, rules
        )
        adapted_lemma = _filter_lemmas(adapted_concept)

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
    related_concept_weight = config.tuning("weight")
    adapted_concepts = {}
    adapted_paths = {}
    all_candidates = {}

    for original_concept, all_shortest_paths in reference_paths.items():
        start_nodes = (
            itertools.chain.from_iterable(rule.target.nodes for rule in rules)
            if config.tuning("bfs", "method") == "within"
            else original_concept.nodes
        )

        adaptation_results = [
            _bfs_adaptation(shortest_path, original_concept, start_nodes)
            for shortest_path in all_shortest_paths
        ]

        adaptation_results = list(itertools.chain.from_iterable(adaptation_results))
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

            _adaptation_candidates[str(candidate)].append(candidate)

        adaptation_candidates = set()
        candidate_occurences = {}
        candidate_length_diff = {}

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
        adapted_lemma = _filter_lemmas(adapted_concept)

        if adapted_concept:
            adapted_concepts[original_concept] = adapted_concept
            log.debug(f"Adapt ({original_concept})->({adapted_concept}).")

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
                    config.tuning("threshold", "synset_similarity", "adaptation"),
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
        filtered_concepts,
        config.tuning("threshold", "concept_score", "adaptation"),
        topn=None,
    )

    if filtered_concepts:
        if occurences and length_differences:
            sorted_concepts = sorted(
                filtered_concepts,
                key=lambda c: 0.75 * c.score
                # + 0.25 * (_dist2sim(length_differences[c]) or 0.0)
                + 0.25 * occurences[c],
                reverse=True,
            )
        else:
            sorted_concepts = sorted(
                filtered_concepts,
                key=lambda c: c.score,
                reverse=True,
            )

        return sorted_concepts[0]

    return None


def _filter_lemmas(
    adapted_concept: t.Optional[casebase.Concept],
) -> t.Optional[casebase.Concept]:
    if adapted_concept is None:
        return None

    lemmas = defaultdict(list)

    for node in adapted_concept.nodes:
        for lemma in node.processed_lemmas:
            lemmas[lemma].append(node)

    if len(lemmas) == 0:
        return adapted_concept

    lemma_vectors = spacy.vectors(lemmas.keys())
    lemma_sim = []
    total_rel_weight = sum(adapted_concept.related_concepts.values())

    for (lemma, nodes), vector in zip(lemmas.items(), lemma_vectors):
        lemma_sim.append(
            (
                lemma,
                vector,
                nodes,
                sum(
                    spacy.similarity(rel_concept.vector, vector) * rel_weight
                    for rel_concept, rel_weight in adapted_concept.related_concepts.items()
                )
                / total_rel_weight,
            )
        )

    lemma_sim.sort(key=lambda x: x[3], reverse=True)
    best_lemma = lemma_sim[0]

    return casebase.Concept.from_concept(
        adapted_concept,
        name=best_lemma[0],
        vector=best_lemma[1],
        metrics=query.concept_metrics(
            adapted_concept.related_concepts,
            adapted_concept.user_query,
            adapted_concept.inodes,
            best_lemma[2],
            best_lemma[1],
            hypernym_proximity=adapted_concept.metrics["hypernym_proximity"],
        ),
    )


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
        spacy.vector(graph.process_name(reference_path.nodes[start_index].name)),
        spacy.vector(graph.process_name(reference_path.nodes[end_index].name)),
        selector,
    )

    for candidate in path_extensions:
        val_adapted = _aggregate_features(
            spacy.vector(graph.process_name(current_path.end_node.name)),
            spacy.vector(graph.process_name(candidate.end_node.name)),
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
