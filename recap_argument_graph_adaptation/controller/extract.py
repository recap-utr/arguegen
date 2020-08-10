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
            if db.node(term):  # original term is in conceptnet
                concepts.add(Concept(term, term))
            elif db.node(lemma):  # lemma is in conceptnet
                concepts.add(Concept(term, lemma))
            else:  # test if the root word is in conceptnet
                root = next(nlp(term).noun_chunks).root

                if db.node(root.text):
                    concepts.add(Concept(term, root.text))

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

                if paths:
                    result[concept] = paths
                    log.info(
                        f"Found {len(paths)} reference paths for ({rule[0]})->({concept})."
                    )
                    log.debug(", ".join((str(path) for path in paths)))
                else:
                    log.info(f"Found 0 reference paths for ({rule[0]})->({concept}).")

    elif method == adaptation.Method.BETWEEN:
        paths = db.all_shortest_paths(rule[0], rule[1])

        if paths:
            _log_paths(paths)

            for concept in concepts:
                if rule[0] != concept.original_name:
                    result[concept] = paths
        else:
            log.error(f"No matching path found. Nothing else to do.")

    else:
        raise ValueError("The parameter 'method' is not set correctly.")

    return result
