import itertools
import statistics
import typing as t
from dataclasses import dataclass

import arguebuf

from arguegen.config import ScoreConfig
from arguegen.model import casebase, wordnet
from arguegen.model.nlp import Nlp


def dist2sim(distance: float) -> float:
    return 1 / (1 + distance)


def weighted_mean(
    values: t.Iterable[float], weights: t.Iterable[float]
) -> t.Optional[float]:
    weights = list(weights)

    try:
        return sum(v * w for v, w in zip(values, weights)) / sum(weights)
    except ZeroDivisionError:
        return None


def filter_concepts(
    scored_concepts: set[casebase.ScoredConcept],
    min_score: float,
    limit: t.Union[None, int, float],
) -> set[casebase.ScoredConcept]:
    filtered_concepts = {entry for entry in scored_concepts if entry.score >= min_score}
    sorted_concepts = sorted(filtered_concepts)

    if isinstance(limit, int) and limit > 0:
        sorted_concepts = sorted_concepts[:limit]
    elif isinstance(limit, float) and 0 < limit < 1:
        sorted_concepts = sorted_concepts[: int(limit * len(sorted_concepts))]

    return set(sorted_concepts)


@dataclass(frozen=True)
class Scorer:
    concept: casebase.Concept
    case_graph: casebase.Graph
    query_graph: casebase.Graph
    related_concepts: tuple[tuple[casebase.Concept, float], ...]
    config: ScoreConfig
    nlp: Nlp
    keyword_weight: t.Optional[float] = None
    hypernym_level: t.Optional[int] = None

    def compute(self) -> float:
        score = 0
        total_weight = 0

        for metric_name, metric_weight in self.config.to_dict().items():
            if (metric := getattr(self, metric_name)) is not None:
                score += metric * metric_weight
                total_weight += metric_weight

        # Normalize the score.
        return score / total_weight

    @property
    def related_concept_weights(self) -> t.Iterable[float]:
        return (weight for _, weight in self.related_concepts)

    @property
    def related_concept_keys(self) -> t.Iterable[casebase.Concept]:
        return (key for key, _ in self.related_concepts)

    @property
    def query_synsets_semantic_similarity(self):
        return wordnet.query_synsets_similarity(
            self.concept.synsets, self.query_graph, self.nlp
        )

    @property
    def query_lemma_semantic_similarity(self):
        return self.nlp.similarity(self.query_graph.text, self.concept.lemma)

    @property
    def query_atoms_semantic_similarity(self):
        return statistics.mean(
            self.nlp.similarities(
                (self.query_graph.text, atom.plain_text) for atom in self.concept.atoms
            )
        )

    @property
    def related_lemmas_semantic_similarity(self):
        sims = self.nlp.similarities(
            (self.concept.lemma, related_concept.lemma)
            for related_concept in self.related_concept_keys
        )
        return weighted_mean(sims, self.related_concept_weights)

    @property
    def related_atoms_semantic_similarity(self):
        return weighted_mean(
            (
                statistics.mean(
                    self.nlp.similarities(
                        (atom1.plain_text, atom2.plain_text)
                        for atom1, atom2 in itertools.product(
                            self.concept.atoms, related_concept.atoms
                        )
                    )
                )
                for related_concept in self.related_concept_keys
            ),
            self.related_concept_weights,
        )

    @property
    def synsets_semantic_similarity(self):
        return weighted_mean(
            (
                wordnet.context_similarity(
                    self.concept.synsets, related_concept.synsets, self.nlp
                )
                for related_concept in self.related_concept_keys
            ),
            self.related_concept_weights,
        )

    @property
    def synsets_path_similarity(self):
        return weighted_mean(
            (
                wordnet.path_similarity(self.concept.synsets, related_concept.synsets)
                for related_concept in self.related_concept_keys
            ),
            self.related_concept_weights,
        )

    @property
    def synsets_wup_similarity(self):
        return weighted_mean(
            (
                wordnet.wup_similarity(self.concept.synsets, related_concept.synsets)
                for related_concept in self.related_concept_keys
            ),
            self.related_concept_weights,
        )

    @property
    def major_claim_proximity(self):
        g = self.case_graph
        mc = g.major_claim or g.root_node
        assert mc is not None

        mc_distances = set()

        for atom in self.concept.atoms:
            if mc_distance := arguebuf.traverse.node_distance(
                atom, mc, g.outgoing_nodes
            ):
                mc_distances.add(mc_distance)

        if mc_distances:
            return dist2sim(min(mc_distances))

        return None

    @property
    def hypernym_proximity(self):
        return dist2sim(self.hypernym_level) if self.hypernym_level else None
