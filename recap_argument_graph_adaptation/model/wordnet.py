from __future__ import annotations

import itertools
import statistics
import typing as t
from collections import defaultdict
from dataclasses import dataclass

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.wordnet import Synset, WordNetCorpusReader
from nltk.corpus.util import LazyCorpusLoader
from recap_argument_graph_adaptation.model import casebase, graph, nlp
from recap_argument_graph_adaptation.model.config import Config

# from nltk.corpus import wordnet as wn

config = Config.instance()
_synset_cache = {}
_synsets_cache = {}


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
    examples: t.Tuple[str, ...]

    @property
    def context(self) -> t.Tuple[str, ...]:
        synset_context = []

        for context_type in config["wordnet"]["node_context_components"]:
            if context_type == "examples":
                synset_context.extend(self.examples)
            elif context_type == "definition":
                synset_context.append(self.definition)

        return tuple(synset_context)

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
                _lemmas=frozenset(lemma.name() for lemma in s.lemmas()),
            )

        raise RuntimeError("The synset does not have a name!")

    def to_nltk(self) -> Synset:
        # with lock:
        if self.uri not in _synset_cache:
            _synset_cache[self.uri] = wn.synset(self.uri)

        return _synset_cache[self.uri]

    def __str__(self) -> str:
        return self.uri

    # @property
    # def name_without_index(self) -> str:
    #     if self.pos:
    #         return ".".join((self.name, self.pos))

    #     return self.name

    def hypernyms(
        self,
        comparison_texts: t.Optional[t.Iterable[str]] = None,
        min_similarity: t.Optional[float] = None,
    ) -> t.FrozenSet[WordnetNode]:
        hyps = frozenset(
            WordnetNode.from_nltk(hyp) for hyp in self.to_nltk().hypernyms()
        )

        if comparison_texts and min_similarity:
            hyps = _filter_nodes(hyps, comparison_texts, min_similarity)

        return hyps

    def hypernym_distances(
        self,
        comparison_texts: t.Optional[t.Iterable[str]] = None,
        min_similarity: t.Optional[float] = None,
    ) -> t.Dict[WordnetNode, int]:
        distances_map = defaultdict(list)

        for hyp_, dist in self.to_nltk().hypernym_distances():
            hyp = WordnetNode.from_nltk(hyp_)

            if (
                hyp != self
                and dist > 0
                and hyp.uri not in config["wordnet"]["hypernym_filter"]
            ):
                distances_map[hyp].append(dist)

        filtered_hypernym_keys = distances_map.keys()

        if comparison_texts and min_similarity:
            filtered_hypernym_keys = _filter_nodes(
                distances_map.keys(), comparison_texts, min_similarity
            )

        return {
            hyp: max(distances)
            for hyp, distances in distances_map.items()
            if hyp in filtered_hypernym_keys
        }

    def metrics(
        self, other: WordnetNode, active: t.Callable[[str], bool]
    ) -> t.Dict[str, t.Optional[float]]:
        synset1 = self.to_nltk()
        synset2 = other.to_nltk()

        path_similarity = (
            (synset1.path_similarity(synset2) or 0.0)
            if active("nodes_path_sim")
            else None
        )
        wup_similarity = (
            (synset1.wup_similarity(synset2) or 0.0)
            if active("nodes_wup_sim")
            else None
        )

        return {
            "nodes_path_sim": path_similarity,
            "nodes_wup_sim": wup_similarity,
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
        return cls.from_nodes(nodes)

    @classmethod
    def from_nodes(cls, nodes: t.Iterable[WordnetNode]) -> WordnetPath:
        nodes = tuple(nodes)
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
        cache_key = (name, pos_tag)

        if cache_key not in _synsets_cache:
            _synsets_cache[cache_key] = wn.synsets(name, pos_tag)

        results.extend(_synsets_cache[cache_key])

    return results


def _nodes_similarities(
    synsets1: t.Iterable[WordnetNode], synsets2: t.Iterable[WordnetNode]
) -> t.List[float]:
    synsets1_ctx = [x.context for x in synsets1]
    synsets2_ctx = [x.context for x in synsets2]
    return list(
        nlp.similarities(
            (ctx1, ctx2)
            for contexts1, contexts2 in itertools.product(synsets1_ctx, synsets2_ctx)
            for ctx1, ctx2 in itertools.product(contexts1, contexts2)
        )
    )


def _filter_nodes(
    synsets: t.AbstractSet[WordnetNode],
    comparison_texts: t.Iterable[str],
    min_similarity: float,
) -> t.FrozenSet[WordnetNode]:
    synsets_contexts = [x.context for x in synsets]
    synset_map = {}

    for synset, synset_contexts in zip(synsets, synsets_contexts):
        similarities = nlp.similarities(
            [
                (x1, x2)
                for x1, x2 in itertools.product(synset_contexts, comparison_texts)
            ]
        )
        synset_map[synset] = statistics.mean(similarities)

    # Check if the best result has a higher similarity than demanded.
    # If true, only include the synsets with higher similarity.
    # Otherwise, include only the best one (regardless of the similarity).
    if synset_map:
        max_similarity = max(synset_map.values())

        if min_similarity < max_similarity:
            synset_map = {
                c: sim for c, sim in synset_map.items() if sim > min_similarity
            }
        else:
            synset_map = {
                c: sim for c, sim in synset_map.items() if sim == max_similarity
            }

    return frozenset(synset_map.keys())


# This function does not use the function _filter_nodes
# Currently only used for determining all shortest paths.
def inherited_hypernyms(node: WordnetNode) -> t.FrozenSet[WordnetPath]:
    hyp_paths = []

    for hyp_path in node.to_nltk().hypernym_paths():
        hyp_sequence = []

        for hyp in reversed(hyp_path[:-1]):  # The last element is the queried node
            hyp_sequence.append(WordnetNode.from_nltk(hyp))

        hyp_paths.append(WordnetPath.from_nodes(hyp_sequence))

    return frozenset(hyp_paths)


def direct_hypernyms(
    node: WordnetNode,
    comparison_texts: t.Iterable[str],
    min_similarity: float,
) -> t.FrozenSet[WordnetPath]:
    hyps = node.hypernyms(comparison_texts, min_similarity)

    return frozenset(WordnetPath.from_nodes((node, hyp)) for hyp in hyps)


def all_shortest_paths(
    start_nodes: t.Iterable[WordnetNode], end_nodes: t.Iterable[WordnetNode]
) -> t.FrozenSet[WordnetPath]:
    all_paths = []

    for start_node, end_node in itertools.product(start_nodes, end_nodes):
        path_candidates = inherited_hypernyms(start_node)

        for path_candidate in path_candidates:
            if end_node in path_candidate.nodes:
                end_idx = path_candidate.nodes.index(end_node)
                shortest_path = (start_node,) + path_candidate.nodes[: end_idx + 1]

                all_paths.append(WordnetPath.from_nodes(shortest_path))

    return frozenset(all_paths)


def concept_synsets(
    names: t.Iterable[str],
    pos: t.Optional[casebase.POS],
    comparison_texts: t.Optional[t.Iterable[str]] = None,
    min_similarity: t.Optional[float] = None,
) -> t.FrozenSet[WordnetNode]:
    synsets = set()

    for name in names:
        synsets.update(
            {
                WordnetNode.from_nltk(ss)
                for ss in _synsets(name, casebase.pos2wn(pos))
                if ss
            }
        )

    synsets = frozenset(synsets)

    if comparison_texts is None or min_similarity is None:
        return synsets

    return _filter_nodes(synsets, comparison_texts, min_similarity)


def metrics(
    synsets1: t.Iterable[WordnetNode],
    synsets2: t.Iterable[WordnetNode],
    active: t.Callable[[str], bool],
) -> t.Dict[str, t.Optional[float]]:
    nodes_semantic_similarity = (
        _nodes_similarities(synsets1, synsets2) if active("nodes_sem_sim") else []
    )

    tmp_results: t.Dict[str, t.MutableSequence[float]] = {
        "nodes_sem_sim": nodes_semantic_similarity,
        "nodes_path_sim": [],
        "nodes_wup_sim": [],
    }

    for s1, s2 in itertools.product(synsets1, synsets2):
        for key, value in s1.metrics(s2, active).items():
            if value is not None:
                tmp_results[key].append(value)

    results: t.Dict[str, t.Optional[float]] = {key: None for key in tmp_results.keys()}

    for key, values in tmp_results.items():
        if values:
            results[key] = statistics.mean(values)

    return results


def query_nodes_similarity(
    synsets: t.Iterable[WordnetNode], user_query: casebase.UserQuery
) -> t.Optional[float]:
    similarities = nlp.similarities(
        (synset.name, user_query.text) for synset in synsets
    )

    return statistics.mean(similarities) if similarities else None
