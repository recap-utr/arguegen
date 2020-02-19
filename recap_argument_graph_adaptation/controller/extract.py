import logging
import typing as t

import recap_argument_graph as ag
import spacy
from textacy import ke

from . import adapt
from ..model.adaptation import Concept
from ..model.config import config
from ..model.database import Database
from ..model import graph, adaptation

log = logging.getLogger(__name__)
nlp = spacy.load(config["spacy"]["model"])


def keywords(graph: ag.Graph, extractor=ke.textrank) -> t.Set[Concept]:
    # ke.textrank, ke.yake, ke.scake, ke.sgrank

    concepts = set()
    db = Database()

    for node in graph.inodes:
        doc = nlp(node.text)

        terms = [key_term for (key_term, weight) in extractor(doc, normalize=None)]
        terms_lemmatized = [key_term for (key_term, weight) in extractor(doc)]

        for term, lemma in zip(terms, terms_lemmatized):
            if db.node(term):
                concepts.add(Concept(term, term))
            elif db.node(lemma):
                concepts.add(Concept(term, lemma))
            else:
                root = next(nlp(term).noun_chunks).root
                if db.node(root.text):
                    concepts.add(Concept(term, root.text))

    return concepts


def paths(
    concepts: t.Iterable[Concept], rule: t.Tuple[str, str]
) -> t.Dict[Concept, t.List[graph.Path]]:
    db = Database()
    result = {}
    method = adaptation.Method(config["adaptation"]["method"])

    log.debug(f"Found the following reference paths:")

    if method == adaptation.Method.WITHIN:
        for concept in concepts:
            if rule[0] != concept.original_name:
                paths = db.all_shortest_paths(rule[0], concept.conceptnet_name)

                if paths:
                    result[concept] = paths
                    _log_paths(paths)

    elif method == adaptation.Method.BETWEEN:
        paths = db.all_shortest_paths(rule[0], rule[1])

        if paths:
            _log_paths(paths)

            for concept in concepts:
                if rule[0] != concept.original_name:
                    result[concept] = paths

    else:
        raise ValueError("The parameter 'method' is not set correctly.")

    return result


def _log_paths(paths: t.Optional[t.Iterable[graph.Path]]):
    if paths:
        log.debug(", ".join((str(path) for path in paths)))
