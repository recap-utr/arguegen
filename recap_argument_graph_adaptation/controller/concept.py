import logging
import typing as t

import recap_argument_graph as ag
import spacy
from textacy import ke

from . import adapt
from ..model.config import config
from ..model.database import Database
from ..model import graph, adaptation

log = logging.getLogger(__package__)
nlp = spacy.load(config["spacy"]["model"])


def from_graph(graph: ag.Graph, extractor=ke.textrank) -> t.Set[str]:
    # extractor is in [ke.textrank,
    #                  ke.yake,
    #                  ke.scake,
    #                  ke.sgrank]
    concepts = set()
    db = Database()

    for node in graph.inodes:
        doc = node.text

        # The keywords are already lemmatized -> FormOf is not necessary
        key_terms = [key_term for (key_term, weight) in extractor(doc)]

        for key_term in key_terms:

            if db.node(key_term):
                concepts.add(key_term)

            else:
                root = next(nlp(key_term).noun_chunks).root
                if db.node(root.text):
                    concepts.add(root.text)

    return concepts


def paths(
    concepts: t.Iterable[str], rule: t.Tuple[str, str]
) -> t.Dict[str, t.Optional[t.List[graph.Path]]]:
    db = Database()
    method = adaptation.Method(config["adaptation"]["method"])

    if method == adaptation.Method.WITHIN:
        return {
            concept: db.all_shortest_paths(rule[0], coencept)
            for concept in concepts
            if rule[0] != concept
        }

    elif method == adaptation.Method.BETWEEN:
        return {
            concept: db.all_shortest_paths(rule[0], rule[1])
            for concept in concepts
            if rule[0] != concept
        }

    raise ValueError("The parameter 'method' is not set correctly.")
