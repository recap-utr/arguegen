from __future__ import annotations

import itertools
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import synchronize

import numpy as np
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.wordnet import Synset as NltkSynset
from nltk.corpus.reader.wordnet import WordNetCorpusReader
from nltk.corpus.util import LazyCorpusLoader
from recap_argument_graph_adaptation.model import casebase, spacy
from recap_argument_graph_adaptation.model.config import Config

# from nltk.corpus import wordnet as wn

config = Config.instance()


def init_reader():
    return LazyCorpusLoader(
        "wordnet",
        WordNetCorpusReader,
        LazyCorpusLoader("omw", CorpusReader, r".*/wn-data-.*\.tab", encoding="utf8"),
    )


wn = init_reader()


# class EmptyLock:
#     def __enter__(self):
#         pass

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         pass


# lock: t.Union[synchronize.Lock, EmptyLock] = EmptyLock()


@dataclass(frozen=True)
class Synset:
    code: str
    definition: str
    examples: t.Tuple[str]

    @classmethod
    def from_nltk(cls, s: NltkSynset) -> Synset:
        return cls(s.name() or "", s.definition() or "", tuple(s.examples()) or tuple())

    def to_nltk(self) -> NltkSynset:
        # with lock:
        return wn.synset(self.code)

    def __str__(self) -> str:
        return self.code

    @property
    def resolved(self) -> t.Tuple[str, casebase.POS]:
        parts = self.code.rsplit(".", 2)
        lemma = parts[0].replace("_", " ")
        pos = casebase.wn_pos_mapping[parts[1]]

        return (lemma, pos)

    @property
    def name_without_index(self) -> str:
        parts = self.code.rsplit(".", 2)[:-1]
        return ".".join(parts)

    def hypernyms(self) -> t.Set[Synset]:
        return {Synset.from_nltk(hyp) for hyp in self.to_nltk().hypernyms()} or set()

    def hypernym_paths(self) -> t.List[t.List[Synset]]:
        # The last element is the current synset and thus can be removed
        return [
            list(reversed([Synset.from_nltk(hyp) for hyp in hyp_path[:-1]]))
            for hyp_path in self.to_nltk().hypernym_paths()
        ]

    def hypernym_distances(self) -> t.Dict[Synset, int]:
        distances_map = defaultdict(list)

        for hyp_, dist in self.to_nltk().hypernym_distances():
            hyp = Synset.from_nltk(hyp_)

            if (
                hyp != self
                and dist > 0
                and hyp.name_without_index not in config["wordnet"]["hypernym_filter"]
            ):
                distances_map[hyp].append(dist)

        return {hyp: max(distances) for hyp, distances in distances_map.items()}

    def compare(self, other: Synset) -> t.Dict[str, float]:
        synset1 = self.to_nltk()
        synset2 = other.to_nltk()

        # with lock:
        return {
            "path_similarity": synset1.path_similarity(synset2) or 0.0,
            "wup_similarity": synset1.wup_similarity(synset2) or 0.0,
        }


def _synsets(
    name: str, pos: t.Union[None, str, t.Collection[str]]
) -> t.List[NltkSynset]:
    name = name.replace(" ", "_")
    results = []
    pos_tags = []

    if not pos or isinstance(pos, str):
        pos_tags.append(pos)
    else:
        pos_tags.extend(pos)

    for pos_tag in pos_tags:
        # with lock:
        new_synsets = wn.synsets(name, pos_tag)

        results.extend(new_synsets)

    return results


def concept_synsets(
    name: str, pos: t.Union[None, str, casebase.POS]
) -> t.Tuple[Synset]:
    if not pos:
        pos_tags = None
    elif isinstance(pos, casebase.POS):
        pos_tags = casebase.wn_pos(pos)
    else:
        pos_tags = [pos]

    return tuple([Synset.from_nltk(ss) for ss in _synsets(name, pos_tags) if ss])


def concept_synsets_contextualized(
    name: str,
    pos: t.Union[None, str, casebase.POS],
    text_vector: np.ndarray,
) -> t.Tuple[Synset, ...]:
    # https://github.com/nltk/nltk/blob/develop/nltk/wsd.py
    synsets = concept_synsets(name, pos)
    synset_definitions = [synset.definition for synset in synsets]
    synset_vectors = spacy.vectors(synset_definitions)
    synset_tuples = [
        (synset, spacy.similarity(text_vector, definition_vector))
        for synset, definition_vector in zip(synsets, synset_vectors)
    ]

    synset_tuples.sort(key=lambda item: item[1])

    # Check if the best result has a higher similarity than demanded.
    # If true, only include the synsets with higher similarity.
    # Otherwise, include only the best one.
    if best_synset_tuple := next(iter(synset_tuples), None):
        min_similarity = config.tuning("synset", "min_similarity")

        if best_synset_tuple[1] > min_similarity:
            synset_tuples = filter(
                lambda x: x[1] > min_similarity,
                synset_tuples,
            )
        else:
            synset_tuples = (best_synset_tuple,)

    return tuple([synset for synset, _ in synset_tuples])


def concept_synset_contextualized(
    name: str,
    pos: t.Union[None, str, casebase.POS],
    text_vector: np.ndarray,
) -> t.Optional[Synset]:
    synsets = concept_synsets_contextualized(name, pos, text_vector)

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
        retrieved_metrics = s1.compare(s2)

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
