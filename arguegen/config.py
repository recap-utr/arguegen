from __future__ import annotations

import typing as t
from dataclasses import dataclass
from enum import Enum

from google.protobuf.struct_pb2 import Struct
from mashumaro import DataClassDictMixin


class BfsMethod(str, Enum):
    WITHIN = "within"
    BETWEEN = "between"


class AdaptationMethod(str, Enum):
    DIRECT = "direct"
    BFS = "bfs"


class SubstitutionMethod(str, Enum):
    TARGET_SCORE = "target_score"
    SOURCE_SCORE = "source_score"
    AGGREGATE_SCORE = "aggregate_score"
    QUERY_SIMILARITY = "query_similarity"


class PruningSelector(str, Enum):
    SIMILARITY = "similarity"
    DIFFERENCE = "difference"


@dataclass
class RelatedConceptWeight(DataClassDictMixin):
    source: float = 0.0
    target: float = 1.0
    original: float = 1.0


@dataclass
class LoaderConfig(DataClassDictMixin):
    heuristic_pos_tags: tuple[str, ...] = ("NOUN", "VERB")
    enforce_node_paths: bool = True
    filter_synsets_based_on_nodes: bool = True


@dataclass
class ExtractionConfig(DataClassDictMixin):
    keyword_pos_tags: tuple[str, ...] = ("NOUN", "VERB")
    keywords_per_adu: bool = False
    concept_limit: t.Union[None, int, float] = None
    synset_similarity_threshold: float = 0.0
    concept_score_threshold: float = 0.0


@dataclass
class AdaptationConfig(DataClassDictMixin):
    lemma_limit: int = 1
    method: AdaptationMethod = AdaptationMethod.DIRECT
    bfs_method: BfsMethod = BfsMethod.BETWEEN
    substitution_method: SubstitutionMethod = SubstitutionMethod.AGGREGATE_SCORE
    related_concept_weight: RelatedConceptWeight = RelatedConceptWeight()
    synset_similarity_threshold: float = 0.0
    concept_score_threshold: float = 0.0
    pruning_selector: PruningSelector = PruningSelector.SIMILARITY
    pruning_bfs_limit: int = 10000


@dataclass
class ScoreConfig(DataClassDictMixin):
    related_atoms_semantic_similarity: float = 0
    related_lemmas_semantic_similarity: float = 0
    keyword_weight: float = 0
    hypernym_proximity: float = 0
    major_claim_proximity: float = 0
    synsets_path_similarity: float = 0
    synsets_semantic_similarity: float = 1
    synsets_wup_similarity: float = 1
    query_atoms_semantic_similarity: float = 0
    query_lemma_semantic_similarity: float = 0
    query_synsets_semantic_similarity: float = 0


@dataclass
class OpenAIConfig(DataClassDictMixin):
    endpoint: str = "chat"


@dataclass
class ExtrasConfig(DataClassDictMixin):
    loader: LoaderConfig = LoaderConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    adaptation: AdaptationConfig = AdaptationConfig()
    score: ScoreConfig = ScoreConfig()
    openai: OpenAIConfig = OpenAIConfig()
    # wordnet, openai, hybrid
    type: str = "wordnet"

    @classmethod
    def from_extras(cls, extras: Struct) -> ExtrasConfig:
        if len(extras) == 0:
            return cls()

        return cls.from_dict(t.cast(t.Mapping, dict(extras.items())))

    def to_extras(self) -> Struct:
        struct = Struct()
        struct.update(self.to_dict())

        return struct
