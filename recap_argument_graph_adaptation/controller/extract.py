import logging
import typing as t

import recap_argument_graph as ag
from recap_argument_graph_adaptation.controller import metrics, spacy

from ..model import adaptation
from ..model.adaptation import Concept, Rule
from ..model.config import config
from ..model.database import Database
from ..model.graph import Path, spacy_pos_mapping
from . import wordnet

log = logging.getLogger(__name__)

spacy_pos_tags = ["NOUN", "PROPN", "VERB", "ADJ"]  # ADV


def keywords(graph: ag.Graph, rules: t.Collection[Rule]) -> t.Set[Concept]:
    related_concepts = {rule.source: 1 / len(rules) for rule in rules}
    rule_sources = {rule.source for rule in rules}
    rule_targets = {rule.target for rule in rules}

    concepts: t.Set[Concept] = set()
    db = Database()

    graph_keywords = spacy.keywords(
        [node.plain_text for node in graph.inodes], spacy_pos_tags
    )
    node_vectors = spacy.vectors([node.plain_text for node in graph.inodes])

    for keywords, node_vector in zip(graph_keywords, node_vectors):
        for k in keywords:
            pos_tag = spacy_pos_mapping[k.pos_tag]
            vector = spacy.vector(k.term)
            nodes = db.nodes(k.term, pos_tag) or db.nodes(k.lemma, pos_tag)
            synsets = wordnet.contextual_synsets(
                node_vector, k.term, pos_tag
            ) or wordnet.contextual_synsets(node_vector, k.lemma, pos_tag)

            if nodes or synsets:
                candidate = Concept(
                    k.term,
                    vector,
                    pos_tag,
                    nodes,
                    synsets,
                    k.weight,
                    *metrics.init_concept_metrics(
                        vector, nodes, synsets, related_concepts
                    ),
                )

                if candidate not in rule_sources and candidate not in rule_targets:
                    concepts.add(candidate)

    concepts = Concept.only_relevant(concepts, config.tuning("extraction", "min_score"))

    log.debug(
        f"Found the following concepts: {', '.join((str(concept) for concept in concepts))}"
    )

    return concepts


def paths(
    concepts: t.Iterable[Concept], rules: t.Collection[adaptation.Rule]
) -> t.Dict[Concept, t.List[Path]]:
    db = Database()
    result = {}
    method = config.tuning("conceptnet", "method")

    if method == "within":
        for concept in concepts:
            paths = []

            for rule in rules:
                if candidates := db.all_shortest_paths(
                    rule.source.nodes, concept.nodes
                ):
                    paths.extend(candidates)

            if paths:
                result[concept] = paths
                log.debug(f"Found {len(paths)} reference path(s) for '{concept}'.")

    elif method == "between":
        paths = []

        for rule in rules:
            if candidates := db.all_shortest_paths(
                rule.source.nodes, rule.target.nodes
            ):
                paths.extend(candidates)

            log.debug(f"Found {len(paths)} reference path(s) for '{rule.target}'.")

        if paths:
            for concept in concepts:
                result[concept] = paths

    else:
        raise ValueError("The parameter 'method' is not set correctly.")

    return result
