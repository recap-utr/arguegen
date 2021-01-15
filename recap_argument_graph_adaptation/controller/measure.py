from __future__ import annotations

import logging
import typing as t

import numpy as np
from recap_argument_graph_adaptation.model import casebase, conceptnet, spacy, wordnet

log = logging.getLogger(__name__)


def _dist2sim(distance: t.Optional[float]) -> t.Optional[float]:
    if distance is not None:
        return 1 / (1 + distance)

    return None


# def update_concept_metrics(
#     concept: casebase.Concept,
#     related_concepts: t.Union[casebase.Concept, t.Mapping[casebase.Concept, float]],
# ) -> t.Tuple[t.Optional[float], ...]:
#     return init_concept_metrics(
#         concept.vector, concept.nodes, concept.synsets, related_concepts
#     )

total_metrics = 5


def init_concept_metrics(
    vector: np.ndarray,
    nodes: t.Sequence[conceptnet.Node],
    synsets: t.Iterable[wordnet.Synset],
    hypernym_level: int,
    related_concepts: t.Union[casebase.Concept, t.Mapping[casebase.Concept, float]],
) -> t.Tuple[t.Optional[float], ...]:
    db = conceptnet.Database()

    if isinstance(related_concepts, casebase.Concept):
        related_concepts = {related_concepts: 1.0}

    if sum(related_concepts.values()) != 1:
        raise ValueError("The weights of the related concepts do not sum up to 1.")

    metrics = [[] for _ in range(total_metrics)]

    for related_concept, weight in related_concepts.items():
        wn_metrics = wordnet.metrics(synsets, related_concept.synsets)
        sim = spacy.similarity(vector, related_concept.vector)

        for i, metric in enumerate(
            (
                sim,
                _dist2sim(hypernym_level),
                _dist2sim(db.distance(nodes, related_concept.nodes)),
                wn_metrics["path_similarity"],
                wn_metrics["wup_similarity"],
            )
        ):
            if metric:
                metrics[i].append(metric * weight)

    # No weight normalization required as given related concepts are available.
    aggregated_metrics = [
        sum(metric_entries) if metric_entries else None for metric_entries in metrics
    ]

    return tuple(aggregated_metrics)
