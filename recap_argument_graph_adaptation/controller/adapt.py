import logging
import multiprocessing
import typing as t
from collections import defaultdict

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
    method: adaptation.Method,
) -> t.Dict[str, str]:
    adapted_concepts = {}

    for concept, all_shortest_paths in reference_paths.items():
        # We are using allShortestPaths, so we pick the one that has the highest number of matches.

        params = [
            (shortest_path, concept, rule, method)
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
    method: adaptation.Method,
) -> graph.Path:
    db = Database()

    # We have to convert the target to a path object here.
    start_name = rule[1] if method == adaptation.Method.WITHIN else concept
    adapted_path = graph.Path.from_node(db.node(start_name))

    for rel in shortest_path.relationships:
        # TODO: The constraint on rel.type is too strict! It could be the case that no matching relation is found.
        path_candidates = db.expand_node(adapted_path.end_node, [rel.type])

        path_candidate = _filter_difference(
            path_candidates, adapted_path, concept, rule
        )

        if path_candidate:
            adapted_path = graph.Path.merge(adapted_path, path_candidate)

    return adapted_path


# TODO: Check if these filters are valid for adaptation 'between'!
def _filter_difference(
    paths: t.Optional[t.Iterable[graph.Path]],
    adapted_path: graph.Path,
    concept: str,
    rule: t.Tuple[str, str],
) -> t.Optional[graph.Path]:
    diff_reference = abs(nlp(concept).vector - nlp(rule[0]).vector)
    candidate_pair = (None, 1.0)
    existing_nodes = set(adapted_path.nodes)

    for path in paths:
        if path.end_node not in existing_nodes:
            diff_adapted = abs(
                nlp(path.end_node.processed_name).vector - nlp(rule[1]).vector
            )
            dist = distance.cosine(diff_reference, diff_adapted)

            if dist < candidate_pair[1]:
                candidate_pair = (path, dist)

    return candidate_pair[0]


def _filter_similarity(
    paths: t.Optional[t.Iterable[graph.Path]],
    adapted_path: graph.Path,
    concept: str,
    rule: t.Tuple[str, str],
) -> t.Optional[graph.Path]:
    dist_reference = distance.cosine(nlp(concept).vector, nlp(rule[0]).vector)
    candidate_pair = (None, 1.0)
    existing_nodes = set(adapted_path.nodes)

    for path in paths:
        if path.end_node not in existing_nodes:
            dist_adapted = distance.cosine(
                nlp(path.end_node.processed_name).vector, nlp(rule[1]).vector
            )
            diff = abs(dist_reference - dist_adapted)

            if diff < candidate_pair[1]:
                candidate_pair = (path, diff)

    return candidate_pair[0]
