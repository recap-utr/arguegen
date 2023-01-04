from __future__ import annotations

import itertools
import statistics
import typing as t
from collections import defaultdict
from dataclasses import dataclass

import wn
import wn.similarity
from wn.morphy import Morphy

from arguegen.config import config
from arguegen.model import casebase, nlp

db = wn.Wordnet("oewn:2021", lemmatizer=Morphy())

_synsets_cache = {}


@dataclass(frozen=True)
class Node:
    # name: str
    # _lemmas: t.FrozenSet[str]
    # pos: t.Optional[str]
    # uri: str
    # index: str
    # definition: str
    # examples: t.Tuple[str, ...]
    # word: wn.Word
    # sense: wn.Sense
    _synset: wn.Synset

    def __eq__(self, other: Node) -> bool:
        return self._synset == other._synset

    def __hash__(self) -> int:
        return hash((self._synset,))

    @property
    def lemmas(self) -> list[wn.Form]:
        return self._synset.lemmas()

    @property
    def pos(self) -> t.Optional[casebase.POS]:
        return casebase.wn2pos(self._synset.pos)

    @property
    def context(self) -> t.Tuple[str, ...]:
        ctx = []

        for context_type in config["wordnet"]["node_context_components"]:
            if context_type == "examples":
                ctx.extend(self._synset.examples())
            elif context_type == "definition" and (
                definition := self._synset.definition()
            ):
                ctx.append(definition)

        return tuple(ctx)

    def __str__(self) -> str:
        return self._synset.id

    def hypernyms(
        self,
        comparison_texts: t.Optional[t.Iterable[str]] = None,
        min_similarity: t.Optional[float] = None,
    ) -> t.FrozenSet[Node]:
        hyps = frozenset(Node(hypernym) for hypernym in self._synset.hypernyms())

        if comparison_texts and min_similarity:
            hyps = _filter_nodes(hyps, comparison_texts, min_similarity)

        return hyps

    def hypernym_distances(
        self,
        comparison_texts: t.Optional[t.Iterable[str]] = None,
        min_similarity: t.Optional[float] = None,
    ) -> t.Dict[Node, int]:
        distances_map: defaultdict[Node, list[int]] = defaultdict(list)

        for path in self._synset.hypernym_paths(
            simulate_root=config.wordnet.simulate_root
        ):
            for dist, hyp_ in enumerate(path):
                hyp = Node(hyp_)

                if (
                    hyp != self
                    and dist > 0
                    and hyp._synset.id not in config["wordnet"]["hypernym_filter"]
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
        self, other: Node, active: t.Callable[[str], bool]
    ) -> t.Dict[str, t.Optional[float]]:
        path_similarity = (
            (wn.similarity.path(self._synset, other._synset))
            if active("nodes_path_sim")
            else None
        )
        wup_similarity = (
            (wn.similarity.wup(self._synset, other._synset))
            if active("nodes_wup_sim")
            else None
        )

        return {
            "nodes_path_sim": path_similarity,
            "nodes_wup_sim": wup_similarity,
        }


@dataclass(frozen=True)
class Relationship:
    type: str
    start_node: Node
    end_node: Node

    @property
    def nodes(self) -> t.Tuple[Node, Node]:
        return (self.start_node, self.end_node)

    def __str__(self):
        return f"{self.start_node}-[{self.type}]->{self.end_node}"

    @classmethod
    def from_nodes(cls, start_node: Node, end_node: Node) -> Relationship:
        return cls(type="Hypernym", start_node=start_node, end_node=end_node)


@dataclass(frozen=True)
class Path:
    nodes: t.Tuple[Node, ...]
    relationships: t.Tuple[Relationship, ...]

    @property
    def start_node(self) -> Node:
        return self.nodes[0]

    @property
    def end_node(self) -> Node:
        return self.nodes[-1]

    def __str__(self):
        out = f"{self.start_node}"

        if len(self.nodes) > 1:
            for node, rel in zip(self.nodes[1:], self.relationships):
                out += f"-[{rel.type}]->{node}"

        return out

    def __len__(self) -> int:
        return len(self.relationships)

    @classmethod
    def from_node(cls, obj: Node) -> Path:
        return cls(nodes=(obj,), relationships=tuple())

    @classmethod
    def merge(cls, *paths: Path) -> Path:
        nodes = paths[0].nodes
        relationships = paths[0].relationships

        for path in paths[1:]:
            nodes += path.nodes[1:]
            relationships += path.relationships

        return cls(
            nodes=nodes,
            relationships=relationships,
        )

    @classmethod
    def from_nodes(cls, nodes: t.Iterable[Node]) -> Path:
        nodes = tuple(nodes)
        relationships = tuple()

        if len(nodes) > 1:
            relationships = tuple(
                (
                    Relationship.from_nodes(nodes[i], nodes[i + 1])
                    for i in range(len(nodes) - 1)
                )
            )

        return cls(nodes=nodes, relationships=relationships)


def _synsets(name: str, pos_tags: t.Collection[t.Optional[str]]) -> t.List[wn.Synset]:
    results = []

    for pos_tag in pos_tags:
        cache_key = (name, pos_tag)

        if cache_key not in _synsets_cache:
            _synsets_cache[cache_key] = wn.synsets(name, pos_tag)  # type: ignore

        results.extend(_synsets_cache[cache_key])

    return results


def _nodes_similarities(
    synsets1: t.Iterable[Node], synsets2: t.Iterable[Node]
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
    synsets: t.AbstractSet[Node],
    comparison_texts: t.Iterable[str],
    min_similarity: float,
) -> t.FrozenSet[Node]:
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
def inherited_hypernyms(node: Node) -> t.FrozenSet[Path]:
    hyp_paths = []

    for hyp_path in node._synset.hypernym_paths(
        simulate_root=config.wordnet.simulate_root
    ):
        hyp_sequence = []

        for hyp in reversed(hyp_path[:-1]):  # The last element is the queried node
            hyp_sequence.append(Node(hyp))

        hyp_paths.append(Path.from_nodes(hyp_sequence))

    return frozenset(hyp_paths)


def direct_hypernyms(
    node: Node,
    comparison_texts: t.Iterable[str],
    min_similarity: float,
) -> t.FrozenSet[Path]:
    hyps = node.hypernyms(comparison_texts, min_similarity)

    return frozenset(Path.from_nodes((node, hyp)) for hyp in hyps)


def all_shortest_paths(
    start_nodes: t.Iterable[Node], end_nodes: t.Iterable[Node]
) -> t.FrozenSet[Path]:
    all_paths = []

    for start_node, end_node in itertools.product(start_nodes, end_nodes):
        path_candidates = inherited_hypernyms(start_node)

        for path_candidate in path_candidates:
            if end_node in path_candidate.nodes:
                end_idx = path_candidate.nodes.index(end_node)
                shortest_path = (start_node,) + path_candidate.nodes[: end_idx + 1]

                all_paths.append(Path.from_nodes(shortest_path))

    if len(all_paths) > 0:
        shortest_length = min(len(path) for path in all_paths)

        return frozenset(path for path in all_paths if len(path) == shortest_length)

    return frozenset()


def concept_synsets(
    names: t.Iterable[str],
    pos: t.Optional[casebase.POS],
    comparison_texts: t.Optional[t.Iterable[str]] = None,
    min_similarity: t.Optional[float] = None,
) -> t.FrozenSet[Node]:
    synsets = set()

    for name in names:
        synsets.update({Node(ss) for ss in _synsets(name, casebase.pos2wn(pos)) if ss})

    synsets = frozenset(synsets)

    if comparison_texts is None or min_similarity is None:
        return synsets

    return _filter_nodes(synsets, comparison_texts, min_similarity)


def metrics(
    synsets1: t.Iterable[Node],
    synsets2: t.Iterable[Node],
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
    synsets: t.Iterable[Node], user_query: casebase.UserQuery
) -> t.Optional[float]:
    similarities = nlp.similarities(
        (lemma, user_query.text) for synset in synsets for lemma in synset.lemmas
    )

    return statistics.mean(similarities) if similarities else None
