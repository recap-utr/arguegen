from __future__ import annotations

import itertools
import json
import typing as t
from collections import defaultdict
from pathlib import Path

import numpy as np

# from nltk.corpus import wordnet as wn
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.wordnet import Synset, WordNetCorpusReader
from nltk.corpus.util import LazyCorpusLoader
from recap_argument_graph_adaptation.controller import spacy

from ..model import graph
from ..model.config import Config

config = Config.instance()


wn = LazyCorpusLoader(
    "wordnet",
    WordNetCorpusReader,
    LazyCorpusLoader("omw", CorpusReader, r".*/wn-data-.*\.tab", encoding="utf8"),
)


def _synset(code: str) -> Synset:
    return wn.synset(code)


def _synsets(name: str, pos: t.Union[None, str, t.Collection[str]]) -> t.List[Synset]:
    name = name.replace(" ", "_")
    results = []
    pos_tags = []

    if not pos or isinstance(pos, str):
        pos_tags.append(pos)
    else:
        pos_tags.extend(pos)

    for pos_tag in pos_tags:
        results.extend(wn.synsets(name, pos_tag))

    return results


def concept_synsets(name: str, pos: t.Union[None, str, graph.POS]) -> t.List[Synset]:
    if not pos:
        pos_tags = None
    elif isinstance(pos, graph.POS):
        pos_tags = graph.wn_pos(pos)
    else:
        pos_tags = [pos]

    return [ss for ss in _synsets(name, pos_tags) if ss]


def synset_definition(synset: Synset) -> str:
    return synset.definition() or ""


def synset_examples(synset: Synset) -> t.List[str]:
    return synset.examples() or []


def synset_name(synset: Synset) -> str:
    return synset.name() or ""


def synset_hypernyms(synset: Synset) -> t.List[Synset]:
    return synset.hypernyms() or []


def synset_metrics(synset1: Synset, synset2: Synset) -> t.Dict[str, float]:
    return {
        "path_similarity": synset1.path_similarity(synset2) or 0.0,
        "wup_similarity": synset1.wup_similarity(synset2) or 0.0,
    }


# DERIVED FUNCTIONS

# TODO: Multiple lemmas are returned, make selection more robust.
def resolve(synset: Synset) -> t.Tuple[str, graph.POS]:
    name = synset_name(synset)
    parts = name.rsplit(".", 2)
    lemma = parts[0]
    pos = graph.wn_pos_mapping[parts[1]]

    return (lemma, pos)


def contextual_synsets(
    text_vector: np.ndarray, term: str, pos: graph.POS
) -> t.Tuple[Synset, ...]:
    # https://github.com/nltk/nltk/blob/develop/nltk/wsd.py
    synsets = concept_synsets(term, pos)
    synset_definitions = [synset_definition(synset) for synset in synsets]
    synset_vectors = spacy.vectors(synset_definitions)

    synset_tuples = []

    for synset, definition_vector in zip(synsets, synset_vectors):
        similarity = spacy.similarity(text_vector, definition_vector)
        synset_tuples.append((synset, similarity))

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


def contextual_synset(
    text_vector: np.ndarray, term: str, pos: graph.POS
) -> t.Optional[Synset]:
    synsets = contextual_synsets(text_vector, term, pos)

    if len(synsets) > 0:
        return synsets[0]

    return None


def metrics(
    synsets1: t.Iterable[Synset], synsets2: t.Iterable[Synset]
) -> t.Dict[str, t.Optional[float]]:
    tmp_results: t.Dict[str, t.List[float]] = {
        "path_similarity": [],
        "wup_similarity": [],
    }

    for s1, s2 in itertools.product(synsets1, synsets2):
        retrieved_metrics = synset_metrics(s1, s2)

        for key, value in retrieved_metrics.items():
            if value:
                tmp_results[key].append(value)

    results: t.Dict[str, t.Optional[float]] = {key: None for key in tmp_results.keys()}

    for key, values in tmp_results.items():
        if values:
            if "distance" in key:
                results[key] = min(values)
            else:
                results[key] = max(values)

    return results


def hypernym_trees(synset: Synset) -> t.List[t.List[Synset]]:
    hypernym_trees = [[synset]]
    has_hypernyms = [True]
    final_hypernym_trees = []

    while any(has_hypernyms):
        has_hypernyms = []
        new_hypernym_trees = []

        for hypernym_tree in hypernym_trees:
            if new_hypernyms := synset_hypernyms(hypernym_tree[-1]):
                has_hypernyms.append(True)

                for new_hypernym in new_hypernyms:
                    new_hypernym_trees.append([*hypernym_tree, new_hypernym])
            else:
                has_hypernyms.append(False)
                final_hypernym_trees.append(hypernym_tree)

        hypernym_trees = new_hypernym_trees

    return final_hypernym_trees


def remove_index(s: str) -> str:
    parts = s.rsplit(".", 2)[:-1]

    return ".".join(parts)


def hypernyms(synset: Synset) -> t.Set[Synset]:
    result = set()
    trees = hypernym_trees(synset)

    for tree in trees:
        # The first synset is the original one, the last always entity
        tree = tree[1:-1]
        # Some synsets are not relevant for generalization
        tree = filter(
            lambda s: remove_index(synset_name(synset))
            not in config["wordnet"]["hypernym_filter"],
            tree,
        )

        result.update(tree)

    return result
