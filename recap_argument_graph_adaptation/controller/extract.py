import logging
from recap_argument_graph_adaptation.controller import metrics
import typing as t

import recap_argument_graph as ag
import spacy
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
    extractor = ke.yake
    # ke.textrank, ke.yake, ke.scake, ke.sgrank

    related_concepts = {rule.source: 1 / len(rules) for rule in rules}
    rule_sources = {rule.source for rule in rules}
    rule_targets = {rule.target for rule in rules}

    concepts: t.Set[Concept] = set()
    db = Database()
    nlp = load.spacy_nlp()

    for spacy_pos_tag in spacy_pos_tags:
        pos_tag = spacy_pos_mapping[spacy_pos_tag]

        for node in graph.inodes:
            doc = nlp(node.plain_text)

            terms = [
                (nlp(key_term), weight)
                for (key_term, weight) in extractor(
                    doc, normalize=None, include_pos=spacy_pos_tag
                )
            ]
            terms_lemmatized = [
                (nlp(key_term), weight)
                for (key_term, weight) in extractor(doc, include_pos=spacy_pos_tag)
            ]

            for (term, term_weight), (lemma, lemma_weight) in zip(
                terms, terms_lemmatized
            ):
                # TODO: Maybe only use term or lemma for synsets to improve performance.
                # Results should be identical.
                nodes = db.nodes(term.text, pos_tag) or db.nodes(lemma.text, pos_tag)
                synsets = wordnet.contextual_synsets(
                    doc, term.text, pos_tag
                ) or wordnet.contextual_synsets(doc, lemma.text, pos_tag)

                # if not (nodes or synsets):  # test if the root word is in conceptnet
                #     term_chunks = next(term.noun_chunks, None)

                #     if term_chunks:
                #         root = term_chunks.root

                #         if not nodes:
                #             nodes = db.nodes(root.text, pos_tag)
                #         if not synsets:
                #             synsets = wordnet.contextual_synsets(
                #                 doc, root.text, pos_tag
                #             )

                if nodes or synsets:
                    candidate = Concept(
                        term,
                        pos_tag,
                        nodes,
                        synsets,
                        term_weight,
                        *metrics.init_concept_metrics(
                            term, nodes, synsets, related_concepts
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
