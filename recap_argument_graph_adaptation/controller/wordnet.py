from __future__ import annotations

import warnings
import itertools
import multiprocessing
import statistics
import typing as t

import requests
from recap_argument_graph_adaptation.controller import load
from spacy.tokens import Doc  # type: ignore
from spacy.util import filter_spans
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset

from ..model import graph
from ..model.config import Config

config = Config.instance()
# session = requests.Session()
lock = multiprocessing.Lock()


# def _url(parts: t.Iterable[str]) -> str:
#     return "/".join(
#         [f"http://{config['wordnet']['host']}:{config['wordnet']['port']}", *parts]
#     )


def _synset(code: str) -> Synset:
    return wn.synset(code)


def _synsets(name: str, pos: t.Optional[str]) -> t.List[Synset]:
    results = wn.synsets(name.replace(" ", "_"))

    if pos:
        results = [ss for ss in results if str(ss.pos()) == pos]

    return results


def _plain_synsets(name: str, pos: t.Optional[str]) -> t.List[str]:
    return [ss.name() for ss in _synsets(name, pos) if ss]  # type: ignore


# def log_synsets(synsets: t.Iterable[str]) -> None:
#     for synset in synsets:
#         print(f"Name:       {synset(s).name()}")
#         print(f"Definition: {synset(s).definition()}")
#         print(f"Examples:   {synset(s).examples()}")
#         print()


def synset_definition(code: str) -> str:
    return _synset(code).definition() or ""


def synset_examples(code: str) -> t.List[str]:
    return _synset(code).examples() or []


def synset_hypernyms(code: str) -> t.List[str]:
    hypernyms = _synset(code).hypernyms()
    return [h.name() for h in hypernyms if h]


def synset_metrics(code1: str, code2: str) -> t.Dict[str, t.Optional[float]]:
    s1 = _synset(code1)
    s2 = _synset(code2)

    return {
        "path_similarity": s1.path_similarity(s2),
        "wup_similarity": s1.wup_similarity(s2),
        "path_distance": s1.shortest_path_distance(s2),
    }


def concept_synsets(name: str, pos: t.Optional[str] = None) -> t.List[str]:
    return _plain_synsets(name, pos)


def resolve(code: str) -> t.Tuple[str, graph.POS]:
    parts = code.split(".")
    name = parts[0].replace("_", " ")
    pos = graph.wn_pos_mapping[parts[1]]

    return (name, pos)


def synsets(name: str, pos: graph.POS) -> t.Tuple[str, ...]:
    with lock:
        # results = session.get(
        #     _url(["concept", name, "synsets"]),
        #     params={"pos": graph.wn_pos(pos)},
        # ).json()
        results = concept_synsets(name, graph.wn_pos(pos))

    return tuple(results)


def contextual_synsets(doc: Doc, term: str, pos: graph.POS) -> t.Tuple[str, ...]:
    # https://github.com/nltk/nltk/blob/develop/nltk/wsd.py
    nlp = load.spacy_nlp()
    results = synsets(term, pos)

    synset_tuples = []

    for synset in results:
        similarity = 0
        with lock:
            # definition = session.get(_url(["synset", synset, "definition"])).text
            definition = synset_definition(synset)

        if definition:
            result_doc = nlp(definition)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                similarity = doc.similarity(result_doc)

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


def contextual_synset(doc: Doc, term: str, pos: graph.POS) -> t.Optional[str]:
    synsets = contextual_synsets(doc, term, pos)

    if synsets:
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
        with lock:
            # retrieved_metrics = session.get(_url(["synset", s1, "metrics", s2])).json()
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
            with lock:
                # new_hypernyms = session.get(
                #     _url(["synset", hypernym_tree[-1], "hypernyms"])
                # ).json()
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
