from __future__ import annotations

import itertools
import statistics
import typing as t
from collections import defaultdict
from dataclasses import dataclass

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.wordnet import Lemma as NltkLemma
from nltk.corpus.reader.wordnet import Synset as NltkSynset
from nltk.corpus.reader.wordnet import WordNetCorpusReader
from nltk.corpus.util import LazyCorpusLoader

from arguegen.model import casebase
from arguegen.model.nlp import Nlp


@dataclass(frozen=True)
class WordnetConfig:
    hypernym_filter: tuple[str, ...] = tuple()
    synset_context: tuple[t.Literal["examples", "definition"], ...] = (
        "examples",
        "definition",
    )
    simulate_root: bool = False


config = WordnetConfig()


class Wordnet:
    nlp: Nlp
    db: WordNetCorpusReader
    _synsets_cache: dict[tuple[str, t.Union[str, None]], list[NltkSynset]] = {}

    def __init__(self, nlp: Nlp):
        self.db = t.cast(
            WordNetCorpusReader,
            LazyCorpusLoader(
                "wordnet",
                WordNetCorpusReader,
                LazyCorpusLoader(
                    "omw", CorpusReader, r".*/wn-data-.*\.tab", encoding="utf8"
                ),
            ),
        )
        self.nlp = nlp

    def _synsets(
        self, name: str, pos_tags: t.Collection[t.Optional[str]]
    ) -> t.List[NltkSynset]:
        results = []

        for pos_tag in pos_tags:
            cache_key = (name, pos_tag)

            if cache_key not in self._synsets_cache:
                self._synsets_cache[cache_key] = t.cast(
                    list[NltkSynset], self.db.synsets(name, pos_tag) or []
                )

            results.extend(self._synsets_cache[cache_key])

        return results

    def concept_synsets(
        self,
        names: t.Iterable[str],
        pos: casebase.Pos.ValueType,
        nlp: Nlp,
        comparison_texts: t.Optional[t.Iterable[str]] = None,
        min_similarity: t.Optional[float] = None,
    ) -> t.FrozenSet[Synset]:
        synsets = set()

        for name in names:
            synsets.update(
                {Synset(ss) for ss in self._synsets(name, casebase.pos2wn(pos)) if ss}
            )

        synsets = frozenset(synsets)

        if comparison_texts is None or min_similarity is None:
            return synsets

        return _filter_nodes(synsets, comparison_texts, min_similarity, nlp)


@dataclass(frozen=True)
class Synset:
    # name: str
    # _lemmas: t.FrozenSet[str]
    # pos: t.Optional[str]
    # uri: str
    # index: str
    # definition: str
    # examples: t.Tuple[str, ...]
    # word: wn.Word
    # sense: wn.Sense
    _synset: NltkSynset

    def __eq__(self, other: Synset) -> bool:
        return self._synset == other._synset

    def __hash__(self) -> int:
        return hash((self._synset,))

    @property
    def lemmas(self) -> list[str]:
        return [
            lemma.name() for lemma in t.cast(list[NltkLemma], self._synset.lemmas())
        ]

    @property
    def lemma(self) -> str:
        return self.lemmas[0]

    @property
    def pos(self) -> casebase.Pos.ValueType:
        return casebase.wn2pos(self._synset.pos())

    @property
    def context(self) -> frozenset[str]:
        ctx = []

        if "examples" in config.synset_context:
            ctx.extend(self._synset.examples() or [])

        if "definition" in config.synset_context and (
            definition := self._synset.definition() or ""
        ):
            ctx.append(definition)

        return frozenset(ctx)

    def __str__(self) -> str:
        return self._synset.name() or ""

    def hypernyms(
        self,
        nlp: Nlp,
        comparison_texts: t.Optional[t.Iterable[str]] = None,
        min_similarity: t.Optional[float] = None,
    ) -> t.FrozenSet[Synset]:
        hyps = frozenset(Synset(hypernym) for hypernym in self._synset.hypernyms())

        if comparison_texts and min_similarity:
            hyps = _filter_nodes(hyps, comparison_texts, min_similarity, nlp)

        return hyps

    def hypernym_distances(
        self,
        nlp: Nlp,
        comparison_texts: t.Optional[t.Iterable[str]] = None,
        min_similarity: t.Optional[float] = None,
    ) -> t.Dict[Synset, int]:
        distances_map: defaultdict[Synset, list[int]] = defaultdict(list)

        for nltk_hyp, dist in self._synset.hypernym_distances(
            simulate_root=config.simulate_root
        ):
            hyp = Synset(nltk_hyp)

            if (
                hyp != self
                and dist > 0
                and hyp._synset.name() not in config.hypernym_filter
            ):
                distances_map[hyp].append(dist)

        filtered_hypernym_keys = distances_map.keys()

        if comparison_texts and min_similarity:
            filtered_hypernym_keys = _filter_nodes(
                distances_map.keys(), comparison_texts, min_similarity, nlp
            )

        return {
            hyp: max(distances)
            for hyp, distances in distances_map.items()
            if hyp in filtered_hypernym_keys
        }


@dataclass(frozen=True)
class Relationship:
    type: str
    start_node: Synset
    end_node: Synset

    @property
    def nodes(self) -> t.Tuple[Synset, Synset]:
        return (self.start_node, self.end_node)

    def __str__(self):
        return f"{self.start_node}-[{self.type}]->{self.end_node}"

    @classmethod
    def from_nodes(cls, start_node: Synset, end_node: Synset) -> Relationship:
        return cls(type="Hypernym", start_node=start_node, end_node=end_node)


@dataclass(frozen=True)
class Path:
    nodes: t.Tuple[Synset, ...]
    relationships: t.Tuple[Relationship, ...]

    @property
    def lemmas(self) -> list[str]:
        return self.end_node.lemmas

    @property
    def lemma(self) -> str:
        return self.end_node.lemma

    @property
    def start_node(self) -> Synset:
        return self.nodes[0]

    @property
    def end_node(self) -> Synset:
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
    def from_node(cls, obj: Synset) -> Path:
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
    def from_nodes(cls, nodes: t.Iterable[Synset]) -> Path:
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


def _filter_nodes(
    synsets: t.AbstractSet[Synset],
    comparison_texts: t.Iterable[str],
    min_similarity: float,
    nlp: Nlp,
) -> t.FrozenSet[Synset]:
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
def inherited_hypernyms(node: Synset) -> t.FrozenSet[Path]:
    hyp_paths = []

    # TODO: simulate_root=config.simulate_root
    for hyp_path in node._synset.hypernym_paths():
        hyp_sequence = []

        for hyp in reversed(hyp_path[:-1]):  # The last element is the queried node
            hyp_sequence.append(Synset(hyp))

        hyp_paths.append(Path.from_nodes(hyp_sequence))

    return frozenset(hyp_paths)


def direct_hypernyms(
    nlp: Nlp,
    node: Synset,
    comparison_texts: t.Iterable[str],
    min_similarity: float,
) -> t.FrozenSet[Path]:
    hyps = node.hypernyms(nlp, comparison_texts, min_similarity)

    return frozenset(Path.from_nodes((node, hyp)) for hyp in hyps)


def all_shortest_paths(
    start_nodes: t.Iterable[Synset], end_nodes: t.Iterable[Synset]
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


def context_similarity(
    synsets1: t.Iterable[Synset], synsets2: t.Iterable[Synset], nlp: Nlp
) -> float:
    contexts1 = itertools.chain.from_iterable(x.context for x in synsets1)
    contexts2 = itertools.chain.from_iterable(x.context for x in synsets2)

    try:
        return statistics.mean(
            nlp.similarities(
                (ctx1, ctx2) for ctx1, ctx2 in itertools.product(contexts1, contexts2)
            )
        )
    except statistics.StatisticsError:
        return 0


def path_similarity(
    synsets1: t.Iterable[Synset], synsets2: t.Iterable[Synset]
) -> float:
    try:
        return statistics.mean(
            (
                t.cast(
                    float,
                    s1._synset.path_similarity(s2._synset, config.simulate_root),
                )
            )
            for s1, s2 in itertools.product(synsets1, synsets2)
        )
    except statistics.StatisticsError:
        return 0


def wup_similarity(synsets1: t.Iterable[Synset], synsets2: t.Iterable[Synset]) -> float:
    try:
        return statistics.mean(
            (
                t.cast(
                    float,
                    s1._synset.wup_similarity(s2._synset, config.simulate_root),
                )
            )
            for s1, s2 in itertools.product(synsets1, synsets2)
        )
    except statistics.StatisticsError:
        return 0


def query_synsets_similarity(
    synsets: t.Iterable[Synset], user_query: casebase.Graph, nlp: Nlp
) -> t.Optional[float]:
    try:
        return statistics.mean(
            nlp.similarities(
                (lemma, user_query.text)
                for synset in synsets
                for lemma in synset.lemmas
            )
        )
    except statistics.StatisticsError:
        return None
