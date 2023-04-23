from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import arguebuf as ag
import immutables
from arg_services.cbr.v1beta import adaptation_pb2
from arg_services.cbr.v1beta.adaptation_pb2 import Pos
from arg_services.cbr.v1beta.model_pb2 import AnnotatedGraph
from nltk.corpus.reader import wordnet as wn

from arguegen.model import wordnet

log = logging.getLogger(__name__)


class Graph(ag.Graph):
    text: str

    def copy(self) -> Graph:
        g = t.cast(Graph, ag.copy(self, config=ag.load.Config(GraphClass=Graph)))
        g.text = self.text

        return g

    @classmethod
    def load(cls, obj: AnnotatedGraph) -> Graph:
        g = t.cast(
            cls, ag.load.protobuf(obj.graph, config=ag.load.Config(GraphClass=Graph))
        )
        g.text = obj.text

        return g

    def dump(self) -> AnnotatedGraph:
        return AnnotatedGraph(graph=ag.dump.protobuf(self), text=self.text)


@dataclass(frozen=True, eq=True)
class ScoredConcept:
    concept: Concept
    score: float

    def dump(self) -> adaptation_pb2.Concept:
        obj = self.concept.dump()
        obj.MergeFrom(adaptation_pb2.Concept(score=self.score))

        return obj


@dataclass(frozen=True, eq=True)
class Concept:
    lemma: str
    form2pos: immutables.Map[str, t.Tuple[str, ...]]
    pos2form: immutables.Map[str, t.Tuple[str, ...]]
    _pos: Pos.ValueType
    atoms: t.FrozenSet[ag.AtomNode]
    synsets: t.FrozenSet[wordnet.Synset] = field(compare=False)

    @property
    def pos(self) -> t.Optional[Pos.ValueType]:
        return None if self._pos == Pos.POS_UNSPECIFIED else self._pos

    @property
    def forms(self) -> t.FrozenSet[str]:
        return frozenset(self.form2pos.keys())

    def __str__(self):
        code = self.code

        if self.atoms:
            code += f"/{set(atom.id for atom in self.atoms)}"

        return code

    def part_eq(self, other: Concept) -> bool:
        return self.pos == other.pos and self.atoms == other.atoms

    @property
    def code(self) -> str:
        out = f"{self.lemma}"

        if self.pos:
            out += f"/{pos2str(self.pos)}"

        return out

    def dump(self) -> adaptation_pb2.Concept:
        return adaptation_pb2.Concept(lemma=self.lemma, pos=self._pos)

    @classmethod
    def from_concept(
        cls,
        source: Concept,
        lemma=None,
        form2pos=None,
        pos2form=None,
        pos=None,
        atoms=None,
        synsets=None,
    ) -> Concept:
        return cls(
            lemma or source.lemma,
            form2pos or source.form2pos,
            pos2form or source.pos2form,
            pos or source._pos,
            atoms or source.atoms,
            synsets or source.synsets,
        )


_Concept = t.TypeVar("_Concept", Concept, ScoredConcept)


@dataclass(frozen=True, eq=True)
class Rule(t.Generic[_Concept]):
    source: _Concept
    target: _Concept

    def __str__(self) -> str:
        return f"({self.source})->({self.target})"

    def dump(self) -> adaptation_pb2.Rule:
        return adaptation_pb2.Rule(source=self.source.dump(), target=self.target.dump())


@dataclass(frozen=True, eq=True)
class Case:
    name: str
    query_graph: Graph
    case_graph: Graph
    rules: t.Tuple[Rule[Concept], ...]

    def __str__(self) -> str:
        return self.name


def spacy2pos(pos: t.Optional[str]) -> Pos.ValueType:
    if pos is None:
        return Pos.POS_UNSPECIFIED

    return {
        "NOUN": Pos.POS_NOUN,
        "PROPN": Pos.POS_NOUN,
        "VERB": Pos.POS_VERB,
        "ADJ": Pos.POS_ADJECTIVE,
        "ADV": Pos.POS_ADVERB,
    }[pos]


def wn2pos(pos: t.Optional[str]) -> Pos.ValueType:
    if pos is None:
        return Pos.POS_UNSPECIFIED

    return {
        wn.NOUN: Pos.POS_NOUN,
        wn.VERB: Pos.POS_VERB,
        wn.ADJ: Pos.POS_ADJECTIVE,
        wn.ADJ_SAT: Pos.POS_ADJECTIVE,
        wn.ADV: Pos.POS_ADVERB,
    }.get(pos, Pos.POS_UNSPECIFIED)


def pos2wn(pos: Pos.ValueType) -> t.List[t.Optional[str]]:
    if pos is None:
        return [None]

    mapping: dict[Pos.ValueType, list[t.Optional[str]]] = {
        Pos.POS_NOUN: [wn.NOUN],
        Pos.POS_VERB: [wn.VERB],
        Pos.POS_ADJECTIVE: [wn.ADJ, wn.ADJ_SAT],
        Pos.POS_ADVERB: [wn.ADV],
    }

    return mapping.get(pos, [None])


def pos2spacy(pos: Pos.ValueType) -> t.List[t.Optional[str]]:
    if pos == Pos.POS_NOUN:
        return ["NOUN"]  # "PROPN"
    elif pos == Pos.POS_VERB:
        return ["VERB"]
    elif pos == Pos.POS_ADJECTIVE:
        return ["ADJ"]
    elif pos == Pos.POS_ADVERB:
        return ["ADV"]

    return [None]


def pos2str(pos: Pos.ValueType) -> str:
    return {
        Pos.POS_NOUN: "noun",
        Pos.POS_VERB: "verb",
        Pos.POS_ADJECTIVE: "adjective",
        Pos.POS_ADVERB: "adverb",
    }[pos]
