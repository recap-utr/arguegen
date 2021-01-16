from __future__ import annotations

import itertools
import typing as t
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.wordnet import Synset, WordNetCorpusReader
from nltk.corpus.util import LazyCorpusLoader
from recap_argument_graph_adaptation.model import casebase, graph, spacy
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
class WordnetNode(graph.AbstractNode):
    index: str
    definition: str
    examples: t.Tuple[str]

    @classmethod
    def from_nltk(cls, s: Synset) -> WordnetNode:
        if code := s.name():
            name, pos, index = tuple(code.rsplit(".", 2))  # type: ignore

            return cls(
                name=name,
                pos=pos,
                index=index,
                uri=code,
                definition=s.definition() or "",
                examples=tuple(s.examples()) or tuple(),
            )

        raise RuntimeError("The synset does not have a name!")

    def to_nltk(self) -> Synset:
        # with lock:
        return wn.synset(self.uri)

    def __str__(self) -> str:
        return self.uri

    @property
    def name_without_index(self) -> str:
        if self.pos:
            return ".".join((self.name, self.pos))

        return self.name

    def hypernyms(self) -> t.FrozenSet[WordnetNode]:
        return (
            frozenset(
                {WordnetNode.from_nltk(hyp) for hyp in self.to_nltk().hypernyms()}
            )
            or frozenset()
        )

    # def hypernym_paths(self) -> t.List[t.List[WordnetNode]]:
    #     # The last element is the current synset and thus can be removed
    #     return [
    #         list(reversed([WordnetNode.from_nltk(hyp) for hyp in hyp_path[:-1]]))
    #         for hyp_path in self.to_nltk().hypernym_paths()
    #     ]

    def hypernym_distances(self) -> t.Dict[WordnetNode, int]:
        distances_map = defaultdict(list)

        for hyp_, dist in self.to_nltk().hypernym_distances():
            hyp = WordnetNode.from_nltk(hyp_)

            if (
                hyp != self
                and dist > 0
                and hyp.name_without_index not in config["wordnet"]["hypernym_filter"]
            ):
                distances_map[hyp].append(dist)

        return {hyp: max(distances) for hyp, distances in distances_map.items()}

    def metrics(self, other: WordnetNode) -> t.Dict[str, float]:
        synset1 = self.to_nltk()
        synset2 = other.to_nltk()

        # with lock:
        return {
            "path_similarity": synset1.path_similarity(synset2) or 0.0,
            "wup_similarity": synset1.wup_similarity(synset2) or 0.0,
        }


@dataclass(frozen=True)
class WordnetRelationship(graph.AbstractRelationship):
    # start_node: Node
    # end_node: Node

    @classmethod
    def from_nltk(cls, start_synset: Synset, end_synset: Synset) -> WordnetRelationship:
        return cls(
            type="Hypernym",
            start_node=WordnetNode.from_nltk(start_synset),
            end_node=WordnetNode.from_nltk(end_synset),
        )

    @classmethod
    def from_nodes(
        cls, start_node: WordnetNode, end_node: WordnetNode
    ) -> WordnetRelationship:
        return cls(type="Hypernym", start_node=start_node, end_node=end_node)


@dataclass(frozen=True)
class WordnetPath(graph.AbstractPath):
    # nodes: t.Tuple[Node, ...]
    # relationships: t.Tuple[Relationship, ...]

    @classmethod
    def from_nltk(cls, synsets: t.Iterable[Synset]) -> WordnetPath:
        nodes = tuple((WordnetNode.from_nltk(synset) for synset in synsets))
        relationships = tuple()

        if len(nodes) > 1:
            relationships = tuple(
                (
                    WordnetRelationship.from_nodes(nodes[i], nodes[i + 1])
                    for i in range(len(nodes) - 1)
                )
            )

        return cls(nodes=nodes, relationships=relationships)


def _synsets(name: str, pos_tags: t.Collection[t.Optional[str]]) -> t.List[Synset]:
    name = name.replace(" ", "_")
    results = []

    for pos_tag in pos_tags:
        # with lock:
        new_synsets = wn.synsets(name, pos_tag)

        results.extend(new_synsets)

    return results


def hypernym_paths(node: WordnetNode) -> t.FrozenSet[WordnetPath]:
    hyp_sequences = [
        list(reversed(hyp_path[:-1])) for hyp_path in node.to_nltk().hypernym_paths()
    ]
    return frozenset(WordnetPath.from_nltk(hyp_seq) for hyp_seq in hyp_sequences)


def concept_synsets(
    name: str,
    pos: t.Optional[casebase.POS],
    text_vector: t.Optional[np.ndarray],
) -> t.FrozenSet[WordnetNode]:
    # https://github.com/nltk/nltk/blob/develop/nltk/wsd.py
    synsets = frozenset(
        {WordnetNode.from_nltk(ss) for ss in _synsets(name, casebase.pos2wn(pos)) if ss}
    )

    if text_vector is None:
        return synsets

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

    return frozenset({synset for synset, _ in synset_tuples})


def metrics(
    synsets1: t.Iterable[WordnetNode], synsets2: t.Iterable[WordnetNode]
) -> t.Dict[str, t.Optional[float]]:
    tmp_results: t.Dict[str, t.List[float]] = {
        "path_similarity": [],
        "wup_similarity": [],
    }

    for s1, s2 in itertools.product(synsets1, synsets2):
        for key, value in s1.metrics(s2).items():
            if value:
                tmp_results[key].append(value)

    results: t.Dict[str, t.Optional[float]] = {key: None for key in tmp_results.keys()}

    for key, values in tmp_results.items():
        if values:
            # if "distance" in key:
            #     results[key] = min(values)
            # else:
            results[key] = max(values)

    return results
