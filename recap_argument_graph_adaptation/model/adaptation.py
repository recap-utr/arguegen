from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field
from enum import Enum
from nltk.corpus.reader.wordnet import Synset

import recap_argument_graph as ag
from recap_argument_graph_adaptation.model import graph
from recap_argument_graph_adaptation.model.config import config
from spacy.tokens import Doc  # type: ignore

log = logging.getLogger(__name__)


class Method(Enum):
    BETWEEN = "between"
    WITHIN = "within"


class Selector(Enum):
    DIFFERENCE = "difference"
    SIMILARITY = "similarity"


@dataclass(frozen=True)
class Concept:
    name: Doc
    pos: graph.POS  # needed as it might be the case that the rule specifies a pos that is not available in ConceptNet.
    nodes: t.Tuple[graph.Node, ...]
    synsets: t.Tuple[Synset, ...]
    semantic_similarity: float
    conceptnet_distance: int
    wordnet_path_similarity: float
    wordnet_wup_similarity: float
    wordnet_path_distance: int

    @property
    def best_node(self) -> graph.Node:
        return self.nodes[0]

    def __str__(self):
        if self.pos != graph.POS.OTHER:
            return f"{self.name.text}/{self.pos.value}"

        return self.name.text

    def __eq__(self, other: Concept) -> bool:
        return self.name.text == other.name.text and self.pos == other.pos

    def __hash__(self) -> int:
        return hash((self.name.text, self.pos))

    @staticmethod
    def only_relevant(concepts: t.Iterable[Concept]) -> t.Set[Concept]:
        kg = config["adaptation"]["knowledge_graph"]
        filter_and = config["nlp"]["filter"]["and"]
        filter_or = config["nlp"]["filter"]["or"]

        if kg == "conceptnet":
            return {
                concept
                for concept in concepts
                if (
                    concept.conceptnet_distance < filter_and["max_conceptnet_distance"]
                    and concept.semantic_similarity
                    > filter_and["min_semantic_similarity"]
                )
                or concept.conceptnet_distance < filter_or["max_conceptnet_distance"]
                or concept.semantic_similarity > filter_or["min_semantic_similarity"]
            }

        elif kg == "wordnet":
            return {
                concept
                for concept in concepts
                if (
                    concept.wordnet_path_distance
                    < filter_and["max_wordnet_path_distance"]
                    and concept.wordnet_path_similarity
                    > filter_and["min_wordnet_path_similarity"]
                    and concept.wordnet_wup_similarity
                    > filter_and["min_wordnet_wup_similarity"]
                    and concept.semantic_similarity
                    > filter_and["min_semantic_similarity"]
                )
                or concept.wordnet_path_distance
                < filter_or["max_wordnet_path_distance"]
                or concept.wordnet_path_similarity
                > filter_or["min_wordnet_path_similarity"]
                or concept.wordnet_wup_similarity
                > filter_or["min_wordnet_wup_similarity"]
                or concept.semantic_similarity > filter_or["min_semantic_similarity"]
            }

        return set()


@dataclass
class Rule:
    source: Concept
    target: Concept

    def __str__(self) -> str:
        return f"({self.source})->({self.target})"


@dataclass(frozen=True)
class Case:
    name: str
    query: str
    graph: ag.Graph
    rules: t.List[Rule]
    benchmark_graph: ag.Graph
    benchmark_rules: t.List[Rule]
