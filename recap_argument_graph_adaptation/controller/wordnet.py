from __future__ import annotations
import itertools
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


def synset(code: str) -> Synset:
    return wn.synset(code)


def log_synsets(synsets: t.Iterable[str]) -> None:
    for s in synsets:
        print(f"Name:       {synset(s).name()}")
        print(f"Definition: {synset(s).definition()}")
        print(f"Examples:   {synset(s).examples()}")
        print()


def resolve_synset(s: str) -> t.Tuple[str, graph.POS]:
    parts = (synset(s).name() or "").split(".")
    name = parts[0].replace("_", " ")
    pos = graph.wn_pos_mapping[parts[1]]

    return (name, pos)


def synsets(term: str, pos: graph.POS) -> t.Tuple[str, ...]:
    results = wn.synsets(term.replace(" ", "_"))

    if pos != graph.POS.OTHER:
        results = (ss for ss in results if str(ss.pos()) == graph.wn_pos(pos))

    return tuple((res.name() for res in results))


def contextual_synsets(doc: Doc, term: str, pos: graph.POS) -> t.Tuple[str, ...]:
    # https://github.com/nltk/nltk/blob/develop/nltk/wsd.py
    nlp = load.spacy_nlp()
    results = synsets(term, pos)

    synset_tuples = []

    for result in results:
        similarity = 0
        s = synset(result)

        if definition := s.definition():
            result_doc = nlp(definition)
            similarity = doc.similarity(result_doc)

        synset_tuples.append((result, similarity))

    synset_tuples.sort(key=lambda item: item[1])

    # Check if the best result has a higher similarity than demanded.
    # If true, only include the synsets with higher similarity.
    # Otherwise, include only the best one.
    if best_synset_tuple := next(iter(synset_tuples), None):
        if best_synset_tuple[1] > config.tuning("hypernym", "min_similarity"):
            synset_tuples = filter(
                lambda x: x[1] > config.tuning("hypernym", "min_similarity"),
                synset_tuples,
            )
        else:
            synset_tuples = (best_synset_tuple,)

    return tuple([synset for synset, _ in synset_tuples])


def contextual_synset(doc: Doc, term: str, pos: graph.POS) -> t.Optional[str]:
    synsets = contextual_synsets(doc, term, pos)

    if synsets:
        return synsets[0]

    return None


def path_similarity(
    synsets1: t.Iterable[str], synsets2: t.Iterable[str]
) -> t.Optional[float]:
    values = []
    ss1 = [synset(s) for s in synsets1]
    ss2 = [synset(s) for s in synsets2]

    for s1, s2 in itertools.product(ss1, ss2):
        if value := s1.path_similarity(s2):
            values.append(value)

    if values:
        return max(values)

    return None


def wup_similarity(
    synsets1: t.Iterable[str], synsets2: t.Iterable[str]
) -> t.Optional[float]:
    values = []
    ss1 = [synset(s) for s in synsets1]
    ss2 = [synset(s) for s in synsets2]

    for s1, s2 in itertools.product(ss1, ss2):
        if value := s1.wup_similarity(s2):
            values.append(value)

    if values:
        return max(values)

    return None


def path_distance(
    synsets1: t.Iterable[str], synsets2: t.Iterable[str]
) -> t.Optional[float]:
    values = []
    ss1 = [synset(s) for s in synsets1]
    ss2 = [synset(s) for s in synsets2]

    for s1, s2 in itertools.product(ss1, ss2):
        if value := s1.shortest_path_distance(s2):
            values.append(value)

    if values:
        return min(values)

    return None


best_metrics = (1, 1, 0)


def hypernym_trees(s: str) -> t.List[t.List[str]]:
    hypernym_trees = [[synset(s)]]
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

    return [[s.name() for s in tree] for tree in final_hypernym_trees]


def remove_index(s: str) -> str:
    parts = s.split(".")[:-1]

    return ".".join(parts)


def hypernyms(s: str) -> t.Set[str]:
    result = set()
    trees = hypernym_trees(s)

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
