import logging
import typing as t

import recap_argument_graph as ag
import spacy
from textacy import ke

from . import adapt, load
from ..model.adaptation import Concept, Rule
from ..model.config import config
from ..model.database import Database
from ..model import adaptation
from ..model.graph import POS, Path, spacy_pos_mapping

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
                nlp(key_term)
                for (key_term, weight) in extractor(
                    doc, normalize=None, include_pos=spacy_pos_tag
                )
            ]
            terms_lemmatized = [
                nlp(key_term)
                for (key_term, weight) in extractor(doc, include_pos=spacy_pos_tag)
            ]

            for term, lemma in zip(terms, terms_lemmatized):
                term_nodes = db.nodes(term.text, pos_tag)
                lemma_nodes = db.nodes(lemma.text, pos_tag)

                relevance = term.similarity(rule.source.name)

                if term_nodes:  # original term is in conceptnet
                    distance = db.distance(term_nodes, rule.source.nodes)
                    concepts.add(
                        Concept(term, pos_tag, term_nodes, relevance, distance)
                    )

                elif lemma_nodes:  # lemma is in conceptnet
                    distance = db.distance(lemma_nodes, rule.source.nodes)
                    concepts.add(
                        Concept(term, pos_tag, lemma_nodes, relevance, distance)
                    )

                else:  # test if the root word is in conceptnet
                    root = next(term.noun_chunks).root
                    root_node = db.nodes(root.text, pos_tag)

                    if root_node:
                        distance = db.distance(root_node, rule.source.nodes)
                        concepts.add(
                            Concept(term, pos_tag, root_node, relevance, distance)
                        )

    # TODO: Make this configurable
    concepts = {
        concept
        for concept in concepts
        if concept.conceptual_distance
        < config["conceptnet"]["nodes"]["max_conceptual_distance"]
        and concept.semantic_similarity
        > config["conceptnet"]["nodes"]["min_semantic_similarity"]
    }

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
                    f"Found {len(paths) if paths else 0} reference paths for ({rule.source})->({concept})."
                )

                if paths:
                    result[concept] = paths
                    log.debug(", ".join((str(path) for path in paths)))

    elif method == adaptation.Method.BETWEEN:
        paths = db.all_shortest_paths(rule.source.nodes, rule.target.nodes)
        log.info(
            f"Found {len(paths) if paths else 0} reference paths for ({rule.source})->({rule.target})."
        )

        if paths:
            for concept in concepts:
                if rule.source != concept:
                    result[concept] = paths

    else:
        raise ValueError("The parameter 'method' is not set correctly.")

    return result
