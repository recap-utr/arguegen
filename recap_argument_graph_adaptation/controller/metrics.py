from __future__ import annotations

import logging
from recap_argument_graph_adaptation.model.adaptation import Concept
import statistics
import typing as t
from dataclasses import dataclass, field
from enum import Enum

import recap_argument_graph as ag
from recap_argument_graph_adaptation.controller import wordnet
from recap_argument_graph_adaptation.model import graph
from recap_argument_graph_adaptation.model.config import config
from recap_argument_graph_adaptation.model.database import Database
from spacy.tokens import Doc  # type: ignore

log = logging.getLogger(__name__)


best_concept_metrics = (1, 0, 1, 1, 0)


def update_concept_metrics(
    concept: Concept, related_concepts: t.Union[Concept, t.Mapping[Concept, float]]
) -> t.Tuple[
    t.Optional[float],
    t.Optional[float],
    t.Optional[float],
    t.Optional[float],
    t.Optional[float],
]:
    return init_concept_metrics(
        concept.name, concept.nodes, concept.synsets, related_concepts
    )


def init_concept_metrics(
    name: Doc,
    nodes: t.Sequence[graph.Node],
    synsets: t.Iterable[str],
    related_concepts: t.Union[Concept, t.Mapping[Concept, float]],
) -> t.Tuple[
    t.Optional[float],
    t.Optional[float],
    t.Optional[float],
    t.Optional[float],
    t.Optional[float],
]:
    db = Database()

    if isinstance(related_concepts, Concept):
        related_concepts = {related_concepts: 1.0}

    if sum(related_concepts.values()) != 1:
        raise ValueError("The weights do not sum up to 1.")

    metrics = [[] for _ in best_concept_metrics]

    for related_concept, weight in related_concepts.items():
        wn_metrics = wordnet.metrics(synsets, related_concept.synsets)

        for i, metric in enumerate(
            (
                name.similarity(related_concept.name),
                db.distance(nodes, related_concept.nodes),
                wn_metrics["path_similarity"],
                wn_metrics["wup_similarity"],
                wn_metrics["path_distance"],
            )
        ):
            if metric:
                metrics[i].append(metric * weight)

    # No weight normalization required as given related concepts are available.
    aggregated_metrics = [
        sum(metric_entries) if metric_entries else None for metric_entries in metrics
    ]

    return tuple(aggregated_metrics)
