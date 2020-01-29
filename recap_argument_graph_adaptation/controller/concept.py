import json
import logging
import multiprocessing
import typing as t
from collections import defaultdict, Counter

import recap_argument_graph as ag
import spacy
from scipy.spatial import distance

from ..model import conceptnet, graph
from ..model.database import Database

log = logging.getLogger(__package__)
nlp = spacy.load("en_core_web_lg")


def from_graph(db: Database, graph: ag.Graph) -> t.Set[str]:
    concepts = set()

    for node in graph.inodes:
        doc = node.text  # nlp(node.text)

        for chunk in doc.noun_chunks:
            chunk_text = chunk.text

            # special case 1: noun chunk is single-word pronoun: them, they, it..
            if len(chunk) == 1 and chunk[0].pos_ == "PRON":
                continue

            # special case 2: preceeding stop word: '[A] death penalty', '[THE] sexual orientation'
            # TODO: Make it more robust (stop words may occur in other positions than 0).
            # TODO: stopwords other than at pos 0 may be relevant "catcher [IN] [THE] rye" to the chunk
            elif len(chunk) > 2 and chunk[0].is_stop:
                chunk_text = chunk[1:].text

            # check if noun chunk is concept in ConceptNet otherwise check if root is concept and add
            if db.node(chunk_text):
                concepts.add(chunk_text)

            elif db.node(chunk.root.text):
                concepts.add(chunk.root.text)

    return concepts


def reference_paths(
    db: Database, concepts: t.Iterable[str], rule: t.Tuple[str, str]
) -> t.Dict[str, t.Optional[t.List[graph.Path]]]:
    return {
        concept: db.all_shortest_paths(rule[0], concept)
        for concept in concepts
        if rule[0] != concept
    }


def adapt_paths(
    db: Database,
    reference_paths: t.Dict[str, t.Optional[t.Iterable[graph.Path]]],
    rule: t.Tuple[str, str],
) -> t.Dict[str, t.Optional[graph.Path]]:
    adapted_paths = {}

    for concept, all_shortest_paths in reference_paths.items():
        # We are using allShortestPaths, so we pick the one that has the highest number of matches.

        params = [
            (shortest_path, concept, rule) for shortest_path in all_shortest_paths
        ]

        with multiprocessing.Pool() as pool:
            shortest_path_adaptations = pool.starmap(adapt_shortest_path, params)

        adaptation_candidates = Counter(shortest_path_adaptations)

        # TODO: Multiple paths with the same count might occur!!
        adapted_paths[concept] = max(
            adaptation_candidates, key=adaptation_candidates.get
        )
        log.info(f"Adapt from {concept} to {adapted_paths[concept].end_node.name}")

    return adapted_paths


def adapt_shortest_path(
    shortest_path: graph.Path, concept: str, rule: t.Tuple[str, str],
):
    db = Database("en")

    # We have to convert the target to a path object here.
    adapted_path = graph.Path.from_node(db.node(rule[1]))

    for rel in shortest_path.relationships:
        path_candidates = db.expand_node(adapted_path.end_node, [rel.type])

        path_candidate = filter_paths(path_candidates, adapted_path, concept, rule)

        if path_candidate:
            adapted_path = graph.Path.merge(adapted_path, path_candidate)

    return adapted_path


def filter_paths(
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
            diff_adapted = abs(nlp(path.end_node.name).vector - nlp(rule[1]).vector)
            dist = distance.cosine(diff_reference, diff_adapted)

            if dist < candidate_pair[1]:
                candidate_pair = (path, dist)

    return candidate_pair[0]


def adapt_graph(
    graph: ag.Graph, adapted_paths: t.Dict[str, t.Optional[graph.Path]]
) -> None:
    for concept, path in adapted_paths.items():
        for node in graph.inodes:
            node.text = node.text.replace(
                concept, conceptnet.adapt_name(path.end_node.name, concept),
            )
