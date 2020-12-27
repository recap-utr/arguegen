import logging

import numpy as np
from recap_argument_graph_adaptation.controller import metrics, spacy
import typing as t

import recap_argument_graph as ag
from textacy import ke
import nltk.wsd

from . import adapt, load
from ..model.adaptation import Concept, Rule
from ..model.config import config
from ..model.database import Database
from ..model import adaptation
from ..model.graph import (
    POS,
    Path,
    spacy_pos_mapping,
    wn_pos,
)
from . import wordnet

log = logging.getLogger(__name__)

spacy_pos_tags = ["NOUN", "PROPN", "VERB", "ADJ", "ADV"]


def keywords(graph: ag.Graph, rules: t.Collection[Rule]) -> t.Set[Concept]:
    related_concepts = {rule.source: 1 / len(rules) for rule in rules}
    rule_sources = {rule.source for rule in rules}
    rule_targets = {rule.target for rule in rules}

    concepts: t.Set[Concept] = set()
    db = Database()

    for node in graph.inodes:
        terms = spacy.keywords(node.plain_text, spacy_pos_tags, False)
        lemmas = spacy.keywords(node.plain_text, spacy_pos_tags, True)

        for (term, _pos_tag, weight), (lemma, _, _) in zip(terms, lemmas):
            pos_tag = spacy_pos_mapping[_pos_tag]
            vector = np.array(spacy.vector(term))
            nodes = db.nodes(term, pos_tag) or db.nodes(lemma, pos_tag)
            synsets = wordnet.contextual_synsets(
                node.plain_text, term, pos_tag
            ) or wordnet.contextual_synsets(node.plain_text, lemma, pos_tag)

            if nodes or synsets:
                candidate = Concept(
                    term,
                    vector,
                    pos_tag,
                    nodes,
                    synsets,
                    weight,
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
