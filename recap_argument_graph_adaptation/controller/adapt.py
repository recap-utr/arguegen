import logging
import multiprocessing
import typing as t
from collections import defaultdict
from enum import Enum

import recap_argument_graph as ag
import spacy
from scipy.spatial import distance

from ..model import conceptnet, graph, adaptation
from ..model.config import config
from ..model.database import Database

log = logging.getLogger(__package__)
nlp = spacy.load(config["spacy"]["model"])


def argument_graph(
    graph: ag.Graph, rule: t.Tuple[str, str], adapted_concepts: t.Dict[str, str]
) -> None:
    for node in graph.inodes:
        # First perform the rule adaptation.
        # Otherwise, 'death penalty' might already be replaced using the adaptation of 'death'.
        node.text = node.text.replace(rule[0], rule[1])

        for concept, adapted_concept in adapted_concepts.items():
            node.text = node.text.replace(concept, adapted_concept,)


def paths(
    reference_paths: t.Mapping[str, t.Optional[t.Iterable[graph.Path]]],
    rule: t.Tuple[str, str],
    adaptation_method: adaptation.Method,
) -> t.Dict[str, str]:
    adapted_concepts = {}

    for root_concept, all_shortest_paths in reference_paths.items():
        # We are using allShortestPaths, so we pick the one that has the highest number of matches.

        params = [
            (shortest_path, root_concept, rule, adaptation_method)
            for shortest_path in all_shortest_paths
        ]

        with multiprocessing.Pool() as pool:
            shortest_path_adaptations = pool.starmap(_adapt_shortest_path, params)

        adaptation_candidates = defaultdict(int)
        reference_length = len(all_shortest_paths[0].relationships)

        for result in shortest_path_adaptations:
            if len(result.relationships) == reference_length:
                adaptation_candidates[result.end_node.processed_name] += 1

        if adaptation_candidates:
            max_value = max(adaptation_candidates.values())
            adapted_names = [
                key
                for key, value in adaptation_candidates.items()
                if value == max_value
            ]
            adapted_name = _filter_concepts(adapted_names, root_concept)
            adapted_name = conceptnet.adapt_name(adapted_name, root_concept)

            adapted_concepts[root_concept] = adapted_name
            log.info(f"Adapt '{root_concept}' to '{adapted_name}'.")

    return adapted_concepts


def _adapt_shortest_path(
    shortest_path: graph.Path,
    concept: str,
    rule: t.Tuple[str, str],
    adaptation_method: adaptation.Method,
) -> graph.Path:
    db = Database()
    selector = adaptation.Selector(config["adaptation"]["selector"])

    # We have to convert the target to a path object here.
    start_name = rule[1] if adaptation_method == adaptation.Method.WITHIN else concept
    adapted_path = graph.Path.from_node(db.node(start_name))

    for rel in shortest_path.relationships:
        path_candidates = db.expand_node(adapted_path.end_node, [rel.type])

        if not path_candidates:  # Relax the relation constraint
            path_candidates = db.expand_node(adapted_path.end_node)

        path_candidate = _filter_paths(
            path_candidates, shortest_path, adapted_path, selector
        )

        if path_candidate:
            adapted_path = graph.Path.merge(adapted_path, path_candidate)

    return adapted_path


def _filter_concepts(adapted_concepts: t.Iterable[str], root_concept: str) -> str:
    root_nlp = nlp(root_concept)
    adapted_concepts_iter = iter(adapted_concepts)

    best_match = (next(adapted_concepts_iter), 0.0)

    for concept in adapted_concepts_iter:
        concept_nlp = nlp(concept)
        sim = root_nlp.similarity(concept_nlp)

        if sim > best_match[1]:
            best_match = (concept, sim)

    return best_match[0]


def _filter_paths(
    candidate_paths: t.Optional[t.Iterable[graph.Path]],
    reference_path: graph.Path,
    adapted_path: graph.Path,
    selector: adaptation.Selector,
) -> t.Optional[graph.Path]:
    end_index = len(adapted_path.nodes)
    start_index = end_index - 1

    val_reference = _aggregate_features(
        nlp(reference_path.nodes[start_index].processed_name).vector,
        nlp(reference_path.nodes[end_index].processed_name).vector,
        selector,
    )

    solution_pair = (None, 1.0)
    existing_nodes = set(adapted_path.nodes)

    for candidate_path in candidate_paths:
        candidate = candidate_path.end_node

        if candidate not in existing_nodes:
            val_adapted = _aggregate_features(
                nlp(adapted_path.nodes[start_index].processed_name).vector,
                nlp(candidate.processed_name).vector,
                selector,
            )
            val_solution = _compare_features(val_reference, val_adapted, selector)

            if val_solution < solution_pair[1]:
                solution_pair = (candidate_path, val_solution)

    return solution_pair[0]


def _aggregate_features(
    feat1: t.Any, feat2: t.Any, selector: adaptation.Selector
) -> t.Any:
    if selector == adaptation.Selector.DIFFERENCE:
        return abs(feat1 - feat2)
    elif selector == adaptation.Selector.SIMILARITY:
        return distance.cosine(feat1, feat2)

    raise ValueError("Parameter 'selector' wrong.")


def _compare_features(
    feat1: t.Any, feat2: t.Any, selector: adaptation.Selector
) -> t.Any:
    if selector == adaptation.Selector.DIFFERENCE:
        return distance.cosine(feat1, feat2)
    elif selector == adaptation.Selector.SIMILARITY:
        return abs(feat1 - feat2)

    raise ValueError("Parameter 'selector' wrong.")
