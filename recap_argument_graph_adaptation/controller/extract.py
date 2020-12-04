import logging
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


def keywords(graph: ag.Graph, rule: Rule) -> t.Set[Concept]:
    extractor = ke.yake
    # ke.textrank, ke.yake, ke.scake, ke.sgrank

    concepts: t.Set[Concept] = set()
    db = Database()
    nlp = load.spacy_nlp()

    for spacy_pos_tag in spacy_pos_tags:
        pos_tag = spacy_pos_mapping[spacy_pos_tag]

        for node in graph.inodes:
            doc = nlp(node.plain_text)

            # TODO: The weight could be used in conjunction with the semantic similarity.
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
                nodes = db.nodes(term.text, pos_tag) or db.nodes(lemma.text, pos_tag)
                synsets = wordnet.contextual_synsets(
                    doc, term.text, pos_tag
                ) or wordnet.contextual_synsets(doc, lemma.text, pos_tag)

                if not nodes and synsets:  # test if the root word is in conceptnet
                    term_chunks = next(term.noun_chunks, None)

                    if term_chunks:
                        root = term_chunks.root
                        nodes = db.nodes(root.text, pos_tag)
                        synsets = wordnet.contextual_synsets(doc, root.text, pos_tag)

                if nodes and synsets:
                    semantic_sim = term.similarity(rule.source.name)
                    wn_metrics = wordnet.metrics(synsets, rule.source.synsets)
                    conceptnet_distance = db.distance(nodes, rule.source.nodes)

                    concepts.add(
                        Concept(
                            term,
                            pos_tag,
                            nodes,
                            synsets,
                            semantic_sim,
                            conceptnet_distance,
                            *wn_metrics,
                            keyword_weight=term_weight,
                        )
                    )

    concepts = Concept.only_relevant(concepts)
    # TODO: Aids wird nicht korrekt erkannt.

    # concept = next(iter(concepts))
    # print(concept.nodes)
    # general = db.nodes_generalizations(concept.nodes)

    log.info(
        f"Found the following concepts: {', '.join((str(concept) for concept in concepts))}"
    )

    return concepts


def paths(
    concepts: t.Iterable[Concept], rule: adaptation.Rule, method: adaptation.Method
) -> t.Dict[Concept, t.List[Path]]:
    db = Database()
    result = {}

    if method == adaptation.Method.WITHIN:
        for concept in concepts:
            if rule.source != concept:
                paths = db.all_shortest_paths(rule.source.nodes, concept.nodes)
                log.info(
                    f"Found {len(paths) if paths else 0} reference path(s) for ({rule.source})->({concept})."
                )

                if paths:
                    result[concept] = paths
                    log.debug(", ".join((str(path) for path in paths)))

    elif method == adaptation.Method.BETWEEN:
        paths = db.all_shortest_paths(rule.source.nodes, rule.target.nodes)
        log.info(
            f"Found {len(paths) if paths else 0} reference path(s) for ({rule.source})->({rule.target})."
        )

        if paths:
            for concept in concepts:
                if rule.source != concept:
                    result[concept] = paths

    else:
        raise ValueError("The parameter 'method' is not set correctly.")

    return result
