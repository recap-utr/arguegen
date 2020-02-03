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


def argument_graph(graph: ag.Graph, adapted_concepts: t.Dict[str, str]) -> None:
    for concept, adapted_concept in adapted_concepts.items():
        for node in graph.inodes:
            node.text = node.text.replace(concept, adapted_concept,)


def paths(
    reference_paths: t.Mapping[str, t.Optional[t.Iterable[graph.Path]]],
    rule: t.Tuple[str, str],
    adaptation_method: adaptation.Method,
) -> t.Dict[str, str]:
    adapted_concepts = {}

    for concept, all_shortest_paths in reference_paths.items():
        # We are using allShortestPaths, so we pick the one that has the highest number of matches.

        params = [
            (shortest_path, concept, rule, adaptation_method)
            for shortest_path in all_shortest_paths
        ]

        with multiprocessing.Pool() as pool:
            shortest_path_adaptations = pool.starmap(_adapt_shortest_path, params)

        adaptation_candidates = defaultdict(int)
        reference_length = len(all_shortest_paths[0].relationships)

        for result in shortest_path_adaptations:
            if len(result.relationships) == reference_length:
                adaptation_candidates[result.end_node.name] += 1

        # TODO: Multiple paths with the same count might occur!!
        if adaptation_candidates:
            adapted_name = max(adaptation_candidates, key=adaptation_candidates.get)
            adapted_name = conceptnet.adapt_name(adapted_name, concept)

            adapted_concepts[concept] = adapted_name
            log.info(f"Adapt '{concept}' to '{adapted_name}'.")

    return adapted_concepts


def _adapt_shortest_path(
    shortest_path: graph.Path,
    concept: str,
    rule: t.Tuple[str, str],
    adaptation_method: adaptation.Method,
) -> graph.Path:
    db = Database()

    # We have to convert the target to a path object here.
    start_name = rule[1] if adaptation_method == adaptation.Method.WITHIN else concept
    adapted_path = graph.Path.from_node(db.node(start_name))

    for rel in shortest_path.relationships:
        # TODO: The constraint on rel.type is too strict! It could be the case that no matching relation is found.
        path_candidates = db.expand_node(adapted_path.end_node, [rel.type])
        filter_method = FilterMethod(config["adaptation"]["filter"])

        path_candidate = _filter(
            path_candidates, shortest_path, adapted_path, filter_method
        )

        if path_candidate:
            adapted_path = graph.Path.merge(adapted_path, path_candidate)

    return adapted_path


class FilterMethod(Enum):
    DIFFERENCE = "difference"
    SIMILARITY = "similarity"


def _filter(
    candidate_paths: t.Optional[t.Iterable[graph.Path]],
    reference_path: graph.Path,
    adapted_path: graph.Path,
    filter_method: FilterMethod,
) -> t.Optional[graph.Path]:
    end_index = len(adapted_path.nodes)
    start_index = end_index - 1

    val_reference = _aggregate_features(
        nlp(reference_path.nodes[start_index].processed_name).vector,
        nlp(reference_path.nodes[end_index].processed_name).vector,
        filter_method,
    )

    solution_pair = (None, 1.0)
    existing_nodes = set(adapted_path.nodes)

    for candidate_path in candidate_paths:
        candidate = candidate_path.end_node

        if candidate not in existing_nodes:
            val_adapted = _aggregate_features(
                nlp(adapted_path.nodes[start_index].processed_name).vector,
                nlp(candidate.processed_name).vector,
                filter_method,
            )
            val_solution = _compare_features(val_reference, val_adapted, filter_method)

            if val_solution < solution_pair[1]:
                solution_pair = (candidate_path, val_solution)

    return solution_pair[0]


def _aggregate_features(
    feat1: t.Any, feat2: t.Any, filter_method: FilterMethod
) -> t.Any:
    if filter_method == FilterMethod.DIFFERENCE:
        return abs(feat1 - feat2)
    elif filter_method == FilterMethod.SIMILARITY:
        return distance.cosine(feat1, feat2)

    raise ValueError("Parameter 'filter_method' wrong.")


def _compare_features(feat1: t.Any, feat2: t.Any, filter_method: FilterMethod) -> t.Any:
    if filter_method == FilterMethod.DIFFERENCE:
        return distance.cosine(feat1, feat2)
    elif filter_method == FilterMethod.SIMILARITY:
        return abs(feat1 - feat2)

    raise ValueError("Parameter 'filter_method' wrong.")
