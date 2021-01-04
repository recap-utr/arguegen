from __future__ import annotations

import logging
import typing as t

import numpy as np
from recap_argument_graph_adaptation.controller import spacy, wordnet
from recap_argument_graph_adaptation.model import graph
from recap_argument_graph_adaptation.model.adaptation import Concept
from recap_argument_graph_adaptation.model.database import Database

log = logging.getLogger(__name__)


best_concept_metrics = (1, 0, 1, 1)


def update_concept_metrics(
    concept: Concept, related_concepts: t.Union[Concept, t.Mapping[Concept, float]]
) -> t.Tuple[t.Optional[float], ...]:
    return init_concept_metrics(
        concept.vector, concept.nodes, concept.synsets, related_concepts
    )


def init_concept_metrics(
    vector: np.ndarray,
    nodes: t.Sequence[graph.Node],
    synsets: t.Iterable[str],
    related_concepts: t.Union[Concept, t.Mapping[Concept, float]],
) -> t.Tuple[t.Optional[float], ...]:
    db = Database()

    if isinstance(related_concepts, Concept):
        related_concepts = {related_concepts: 1.0}

    if sum(related_concepts.values()) != 1:
        raise ValueError("The weights of the related concepts do not sum up to 1.")

    metrics = [[] for _ in best_concept_metrics]

    for related_concept, weight in related_concepts.items():
        wn_metrics = wordnet.metrics(synsets, related_concept.synsets)
        sim = spacy.similarity(vector, related_concept.vector)

        for i, metric in enumerate(
            (
                sim,
                db.distance(nodes, related_concept.nodes),
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
