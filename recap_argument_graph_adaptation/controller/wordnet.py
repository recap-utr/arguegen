from __future__ import annotations

import warnings
import itertools
import multiprocessing
import statistics
import typing as t
import numpy as np

import requests
from recap_argument_graph_adaptation.controller import load, spacy
from spacy.tokens import Doc  # type: ignore
from spacy.util import filter_spans
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset

from ..model import graph
from ..model.config import Config

config = Config.instance()
session = requests.Session()
lock = multiprocessing.Lock()


def _url(parts: t.Iterable[str]) -> str:
    return "/".join(
        [f"http://{config['wordnet']['host']}:{config['wordnet']['port']}", *parts]
    )


# WORDNET API


def _synset(code: str) -> Synset:
    return wn.synset(code)


def _synsets(name: str, pos: t.Optional[str]) -> t.List[Synset]:
    results = wn.synsets(name.replace(" ", "_"))

    if pos:
        results = [ss for ss in results if str(ss.pos()) == pos]

    return results


def concept_synsets(name: str, pos: t.Union[None, str, graph.POS]) -> t.List[str]:
    if pos and isinstance(pos, graph.POS):
        pos = graph.wn_pos(pos)

    return [ss.name() for ss in _synsets(name, pos) if ss]  # type: ignore


def synset_definition(code: str) -> str:
    synset = _synset(code)

    with lock:
        definition = synset.definition() or ""

    return definition

    # return session.get(_url(["synset", synset, "definition"])).text


def synset_examples(code: str) -> t.List[str]:
    synset = _synset(code)

    with lock:
        examples = synset.examples() or []

    return examples


def synset_hypernyms(code: str) -> t.List[str]:
    synset = _synset(code)

    with lock:
        hypernyms = synset.hypernyms()

    return [h.name() for h in hypernyms if h]

    # return session.get(
    #     _url(["synset", synset, "hypernyms"])
    # ).json()


def synset_metrics(code1: str, code2: str) -> t.Dict[str, t.Optional[float]]:
    s1 = _synset(code1)
    s2 = _synset(code2)

    with lock:
        result = {
            "path_similarity": s1.path_similarity(s2),
            "wup_similarity": s1.wup_similarity(s2),
            "path_distance": s1.shortest_path_distance(s2),
        }

    return result

    # return session.get(_url(["synset", code1, "metrics", code2])).json()


# DERIVED FUNCTIONS


def resolve(code: str) -> t.Tuple[str, graph.POS]:
    parts = code.split(".")
    name = parts[0].replace("_", " ")
    pos = graph.wn_pos_mapping[parts[1]]

    return (name, pos)


def contextual_synsets(text: str, term: str, pos: graph.POS) -> t.Tuple[str, ...]:
    # https://github.com/nltk/nltk/blob/develop/nltk/wsd.py
    results = concept_synsets(term, pos)

    synset_tuples = []

    for synset in results:
        similarity = 0

        if synset_def := synset_definition(synset):
            similarity = spacy.similarity(text, synset_def)

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


def contextual_synset(text: str, term: str, pos: graph.POS) -> t.Optional[str]:
    synsets = contextual_synsets(text, term, pos)

    if len(synsets) > 0:
        return synsets[0]

    return None


def metrics(
    synsets1: t.Iterable[str], synsets2: t.Iterable[str]
) -> t.Dict[str, t.Optional[float]]:
    tmp_results: t.Dict[str, t.List[float]] = {
        "path_similarity": [],
        "wup_similarity": [],
        "path_distance": [],
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


best_metrics = (1, 1, 0)


def hypernym_trees(synset: str) -> t.List[t.List[str]]:
    hypernym_trees = [[synset]]
    has_hypernyms = [True]
    final_hypernym_trees = []

    while any(has_hypernyms):
        has_hypernyms = []
        new_hypernym_trees = []

        for hypernym_tree in hypernym_trees:
            new_hypernyms = synset_hypernyms(hypernym_tree[-1])

            if new_hypernyms:
                has_hypernyms.append(True)

                for new_hypernym in new_hypernyms:
                    new_hypernym_trees.append([*hypernym_tree, new_hypernym])
            else:
                has_hypernyms.append(False)
                final_hypernym_trees.append(hypernym_tree)

        hypernym_trees = new_hypernym_trees

    return final_hypernym_trees


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
