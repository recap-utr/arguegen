from __future__ import annotations


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
    name = parts[0]
    pos = graph.wn_pos_mapping[parts[1]]

    return (name, pos)


def synset(code: str) -> Synset:
    return wn.synset(code)


def synsets(term: str, pos: t.Optional[graph.POS]) -> t.Tuple[Synset, ...]:
    results = wn.synsets(term.replace(" ", "_"))

    if pos:
        results = (ss for ss in results if str(ss.pos()) == graph.wn_pos(pos))

    return tuple(results)


def contextual_synset(
    doc: Doc, term: str, pos: t.Optional[graph.POS]
) -> t.Optional[Synset]:
    # https://github.com/nltk/nltk/blob/develop/nltk/wsd.py
    nlp = load.spacy_nlp()
    results = synsets(term, pos)

    if not results:
        return None

    synset_tuples = []

    for result in results:
        similarity = 0

        if definition := result.definition():
            result_doc = nlp(definition)
            similarity = doc.similarity(result_doc)

        synset_tuples.append((similarity, result))

    _, sense = max(synset_tuples)

    return sense


def contextual_synsets(
    doc: Doc, term: str, pos: t.Optional[graph.POS]
) -> t.Tuple[Synset, ...]:
    result = contextual_synset(doc, term, pos)

    if result:
        return (result,)

    return tuple()


def metrics(
    synsets1: t.Iterable[Synset], synsets2: t.Iterable[Synset]
) -> t.Tuple[float, float, int]:
    path_similarities = []
    wup_similarities = []
    path_distances = []

    for s1 in synsets1:
        for s2 in synsets2:
            path_similarities.append(s1.path_similarity(s2))
            wup_similarities.append(s1.wup_similarity(s2))
            path_distances.append(
                s1.shortest_path_distance(s2) or config["nlp"]["max_distance"]
            )

    return (max(path_similarities), max(wup_similarities), min(path_distances))


best_metrics = (1, 1, 0)


def hypernyms(synset: Synset) -> t.List[t.List[Synset]]:
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
