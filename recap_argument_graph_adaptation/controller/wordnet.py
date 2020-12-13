from __future__ import annotations
import statistics

from spacy.util import filter_spans


from recap_argument_graph_adaptation.controller import load

import typing as t
from dataclasses import dataclass
from enum import Enum
from nltk.corpus.reader.wordnet import Synset, wup_similarity
from nltk.corpus import wordnet as wn

from ..model import graph
from ..model.config import Config

from spacy.tokens import Doc  # type: ignore

import neo4j.data as neo4j

config = Config.instance()


def log_synsets(synsets: t.Iterable[Synset]) -> None:
    for synset in synsets:
        print(f"Name:       {synset.name()}")
        print(f"Definition: {synset.definition()}")
        print(f"Examples:   {synset.examples()}")
        print()


def resolve_synset(synset: Synset) -> t.Tuple[str, graph.POS]:
    parts = (synset.name() or "").split(".")
    name = parts[0].replace("_", " ")
    pos = graph.wn_pos_mapping[parts[1]]

    return (name, pos)


def synset(code: str) -> Synset:
    return wn.synset(code)


def synsets(term: str, pos: graph.POS) -> t.Tuple[Synset, ...]:
    results = wn.synsets(term.replace(" ", "_"))

    if pos != graph.POS.OTHER:
        results = (ss for ss in results if str(ss.pos()) == graph.wn_pos(pos))

    return tuple(results)


def contextual_synsets(doc: Doc, term: str, pos: graph.POS) -> t.Tuple[Synset, ...]:
    # https://github.com/nltk/nltk/blob/develop/nltk/wsd.py
    nlp = load.spacy_nlp()
    results = synsets(term, pos)

    synset_tuples = []

    for result in results:
        similarity = 0

        if definition := result.definition():
            result_doc = nlp(definition)
            similarity = doc.similarity(result_doc)

        synset_tuples.append((result, similarity))

    synset_tuples.sort(key=lambda item: item[1])

    # Check if the best result has a higher similarity than demanded.
    # If true, only include the synsets with higher similarity.
    # Otherwise, include only the best one.
    if best_synset_tuple := next(iter(synset_tuples), None):
        if best_synset_tuple[1] > config["wordnet"]["min_similarity_hypernym"]:
            synset_tuples = filter(
                lambda x: x[1] > config["wordnet"]["min_similarity_hypernym"],
                synset_tuples,
            )
        else:
            synset_tuples = (best_synset_tuple,)

    return tuple([synset for synset, _ in synset_tuples])


def contextual_synset(
    doc: Doc, term: str, pos: t.Optional[graph.POS]
) -> t.Optional[Synset]:
    synsets = contextual_synsets(doc, term, pos)

    if synsets:
        return synsets[0]

    return None


def path_similarity(
    synsets1: t.Iterable[Synset], synsets2: t.Iterable[Synset]
) -> t.Optional[float]:
    values = []

    for s1 in synsets1:
        for s2 in synsets2:
            if value := s1.path_similarity(s2):
                values.append(value)

    if values:
        return max(values)

    return None


def wup_similarity(
    synsets1: t.Iterable[Synset], synsets2: t.Iterable[Synset]
) -> t.Optional[float]:
    values = []

    for s1 in synsets1:
        for s2 in synsets2:
            if value := s1.wup_similarity(s2):
                values.append(value)

    if values:
        return max(values)

    return None


def path_distance(
    synsets1: t.Iterable[Synset], synsets2: t.Iterable[Synset]
) -> t.Optional[float]:
    values = []

    for s1 in synsets1:
        for s2 in synsets2:
            if value := s1.shortest_path_distance(s2):
                values.append(value)

    if values:
        return min(values)

    return None


best_metrics = (1, 1, 0)


def hypernym_trees(synset: Synset) -> t.List[t.List[Synset]]:
    hypernym_trees = [[synset]]
    has_hypernyms = [True]
    final_hypernym_trees = []

    while any(has_hypernyms):
        has_hypernyms = []
        new_hypernym_trees = []

        for hypernym_tree in hypernym_trees:
            new_hypernyms = hypernym_tree[-1].hypernyms()

            if new_hypernyms:
                has_hypernyms.append(True)

                for new_hypernym in new_hypernyms:
                    new_hypernym_trees.append([*hypernym_tree, new_hypernym])
            else:
                has_hypernyms.append(False)
                final_hypernym_trees.append(hypernym_tree)

        hypernym_trees = new_hypernym_trees

    return final_hypernym_trees


def remove_index(synset: Synset) -> str:
    name = synset.name() or ""
    parts = name.split(".")[:-1]

    return ".".join(parts)


def hypernyms(synset: Synset) -> t.Set[Synset]:
    result = set()
    trees = hypernym_trees(synset)

    for tree in trees:
        # The first synset is the original one, the last always entity
        tree = tree[1:-1]
        # Some synsets are not relevant for generalization
        tree = filter(
            lambda s: remove_index(s) not in config["wordnet"]["hypernym_filter"],
            tree,
        )

        result.update(tree)

    return result
