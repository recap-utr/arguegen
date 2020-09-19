import logging
import typing as t

import recap_argument_graph as ag
import spacy
from textacy import ke

from . import adapt, load
from ..model.adaptation import Concept
from ..model.config import config
from ..model.database import Database
from ..model import graph, adaptation

log = logging.getLogger(__name__)


def keywords(graph: ag.Graph) -> t.Set[Concept]:
    extractor = ke.yake
    # ke.textrank, ke.yake, ke.scake, ke.sgrank

    concepts = set()
    db = Database()
    nlp = load.spacy_nlp()

    for node in graph.inodes:
        doc = nlp(node.plain_text)

        terms = [key_term for (key_term, weight) in extractor(doc, normalize=None)]
        terms_lemmatized = [key_term for (key_term, weight) in extractor(doc)]

        for term, lemma in zip(terms, terms_lemmatized):
            term_node = db.node(term)
            lemma_node = db.node(lemma)

            if term_node:  # original term is in conceptnet
                concepts.add(Concept(term, term_node.name))

            elif lemma_node:  # lemma is in conceptnet
                concepts.add(Concept(term, lemma_node.name))

            else:  # test if the root word is in conceptnet
                root = next(nlp(term).noun_chunks).root
                root_node = db.node(root.text)

                if root_node:
                    concepts.add(Concept(term, root_node.name))

    return concepts


def paths(
    concepts: t.Iterable[Concept], rule: t.Tuple[str, str], method: adaptation.Method
) -> t.Dict[Concept, t.List[graph.Path]]:
    db = Database()
    result = {}

    log.debug(f"Found the following reference paths:")

    if method == adaptation.Method.WITHIN:
        for concept in concepts:
            if rule[0] != concept.original_name:
                paths = db.all_shortest_paths(rule[0], concept.conceptnet_name)
                log.info(
                    f"Found {len(paths) if paths else 0} reference paths for ({rule[0]})->({concept})."
                )

                if paths:
                    result[concept] = paths
                    log.debug(", ".join((str(path) for path in paths)))

    elif method == adaptation.Method.BETWEEN:
        paths = db.all_shortest_paths(rule[0], rule[1])
        log.info(
            f"Found {len(paths) if paths else 0} reference paths for ({rule[0]})->({rule[1]})."
        )

        if paths:
            for concept in concepts:
                if rule[0] != concept.original_name:
                    result[concept] = paths

    else:
        raise ValueError("The parameter 'method' is not set correctly.")

    return result
