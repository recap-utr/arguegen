import math
import typing as t
from collections import defaultdict

import neo4j
import recap_argument_graph as ag
from scipy.spatial import distance
import spacy

from ..model import conceptnet
from ..model.database import Database

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
) -> t.Dict[str, t.Optional[t.List[neo4j.Path]]]:
    return {
        concept: db.all_shortest_paths(rule[0], concept)
        for concept in concepts
        if rule[0] != concept
    }


def adapt_paths(
    db: Database,
    reference_paths: t.Dict[str, t.Optional[t.Iterable[neo4j.Path]]],
    rule: t.Tuple[str, str],
) -> t.Dict[str, t.Optional[neo4j.Path]]:
    adapted_paths = {}

    for concept, paths in reference_paths.items():
        # We are using allShortestPaths, so we pick the one that has the highest number of matches.
        adaptation_candidates = defaultdict(int)

        for path in paths:
            # We have to convert the target to a path object here.
            adapted_path = db.single_path(rule[1])

            for rel in path:
                path_candidates = db.expand_node(adapted_path.end_node, [rel.type])

                path_candidate = filter_paths(
                    path_candidates, adapted_path, concept, rule
                )

                if path_candidate:
                    adapted_path = db.extend_path(adapted_path, path_candidate)

            adaptation_candidates[adapted_path] += 1

        # TODO: Multiple paths with the same count might occur!!
        adapted_paths[concept] = max(
            adaptation_candidates, key=adaptation_candidates.get
        )

    return adapted_paths


def filter_paths(
    paths: t.Optional[t.Iterable[neo4j.Path]],
    adapted_path: neo4j.Path,
    concept: str,
    rule: t.Tuple[str, str],
) -> t.Optional[neo4j.Path]:
    diff_reference = abs(nlp(concept).vector - nlp(rule[0]).vector)
    candidate_pair = (None, 1.0)
    adapted_node_ids = [node.id for node in adapted_path.nodes]

    for path in paths:
        if path.end_node.id not in adapted_node_ids:
            diff_adapted = abs(nlp(path.end_node["name"]).vector - nlp(rule[1]).vector)
            dist = distance.cosine(diff_reference, diff_adapted)

            if dist < candidate_pair[1]:
                candidate_pair = (path, dist)

    return candidate_pair[0]


def adapt_graph(
    graph: ag.Graph, adapted_paths: t.Dict[str, t.Optional[neo4j.Path]]
) -> None:
    for concept, path in adapted_paths.items():
        for node in graph.inodes:
            node.text = node.text.replace(
                concept, conceptnet.adapt_name(path.end_node["name"], concept),
            )
