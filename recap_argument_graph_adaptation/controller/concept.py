import math
import typing as t

import neo4j
import recap_argument_graph as ag
import spacy
import numpy as np

from ..model import conceptnet
from ..model.database import Database

nlp = spacy.load("en_core_web_lg")


def from_graph(graph: ag.Graph) -> t.Set[str]:
    concepts = set()

    for node in graph.inodes:
        for token in node.text:
            if token.tag_.startswith("N") and not token.is_stop:
                concepts.add(token.text)

        # for chunk in node.text.noun_chunks:
        #     concepts.add(chunk.root.text)

    return concepts


def reference_paths(
    db: Database, concepts: t.Iterable[str], rule: t.Tuple[str, str]
) -> t.Dict[str, t.Optional[neo4j.Path]]:
    return {
        concept: db.shortest_path(rule[0], rule[1])
        for concept in concepts
        if rule[0] != concept
    }


def adapt_paths(
    db: Database,
    reference_paths: t.Dict[str, t.Optional[neo4j.Path]],
    rule: t.Tuple[str, str],
) -> t.Dict[str, t.Optional[neo4j.Path]]:
    adapted_paths = {}

    for concept, reference_path in reference_paths.items():
        # We have to convert the target to a path object here.
        adapted_path = db.single_path(rule[1])

        for reference_rel in reference_path:
            adapted_relationships = db.expand_node(
                adapted_path.end_node, [reference_rel.type]
            )

            adapted_relationship = filter_relationships(
                adapted_relationships, concept, rule
            )
            # adapted_relationship = next(iter(adapted_relationships), None)

            if adapted_relationship:
                adapted_path = db.extend_path(adapted_path, adapted_relationship)

        adapted_paths[concept] = adapted_path

    return adapted_paths


def filter_relationships(
    rels: t.Optional[t.Iterable[neo4j.Relationship]],
    concept: str,
    rule: t.Tuple[str, str],
) -> t.Optional[neo4j.Relationship]:
    diff_reference = abs(nlp(concept).vector - nlp(rule[0]).vector)
    candidate_pair = (None, 1)

    for rel in rels:
        diff_adapted = abs(nlp(rel.end_node["name"]).vector - nlp(rule[1]).vector)
        dist = scipy.spatial.distance.cosine(diff_reference, diff_adapted)

        if dist < candidate_pair[1]:
            candidate_pair = (rel, dist)

    return candidate_pair[0]


def adapt_graph(
    graph: ag.Graph, adapted_paths: t.Dict[str, t.Optional[neo4j.Path]]
) -> None:
    for concept, path in adapted_paths.items():
        for node in graph.inodes:
            node.text = node.text.replace(
                concept, conceptnet.adapt_name(path.end_node["name"], concept),
            )
