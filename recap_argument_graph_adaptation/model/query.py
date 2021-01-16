import typing as t

import numpy as np
from recap_argument_graph_adaptation.model import (
    casebase,
    conceptnet,
    graph,
    spacy,
    wordnet,
)
from recap_argument_graph_adaptation.model.config import Config

config = Config.instance()
_kg = config["adaptation"]["knowledge_graph"]
kg_wn = _kg == "wordnet"
kg_cn = _kg == "conceptnet"


def _dist2sim(distance: t.Optional[float]) -> t.Optional[float]:
    if distance is not None:
        return 1 / (1 + distance)

    return None


def pos(tag: t.Optional[str]) -> t.Optional[casebase.POS]:
    if kg_wn:
        return casebase.wn2pos(tag)

    elif kg_cn:
        return casebase.cn2pos(tag)

    return None


def concept_nodes(
    name: str, pos: t.Optional[casebase.POS], text_vector: t.Optional[np.ndarray] = None
) -> t.FrozenSet[graph.AbstractNode]:
    if kg_wn:
        return wordnet.concept_synsets(name, pos, text_vector)

    elif kg_cn:
        return conceptnet.Database().nodes(name, pos)

    return frozenset()


def concept_metrics(
    nodes: t.Iterable[graph.AbstractNode],
    vector: np.ndarray,
    weight: t.Optional[float],
    hypernym_level: t.Optional[int],
    related_concepts: t.Union[casebase.Concept, t.Mapping[casebase.Concept, float]],
) -> t.Dict[str, t.Optional[float]]:
    if isinstance(related_concepts, casebase.Concept):
        related_concepts = {related_concepts: 1.0}

    if sum(related_concepts.values()) != 1:
        raise ValueError("The weights of the related concepts do not sum up to 1.")

    metrics_map = {key: [] for key in casebase.metric_keys}

    for related_concept, related_concept_weight in related_concepts.items():
        metrics = {
            "keyword_weight": weight,
            "semantic_similarity": spacy.similarity(vector, related_concept.vector),
            "hypernym_proximity": _dist2sim(hypernym_level),
            "path_similarity": None,
            "wup_similarity": None,
        }

        assert metrics.keys() == casebase.metric_keys

        if kg_wn:
            nodes = t.cast(t.Iterable[wordnet.WordnetNode], nodes)
            related_nodes = t.cast(
                t.Iterable[wordnet.WordnetNode], related_concept.nodes
            )
            metrics.update(wordnet.metrics(nodes, related_nodes))

        elif kg_cn:
            db = conceptnet.Database()
            nodes = t.cast(t.Iterable[conceptnet.ConceptnetNode], nodes)
            related_nodes = t.cast(
                t.Iterable[conceptnet.ConceptnetNode], related_concept.nodes
            )
            metrics.update(db.metrics(nodes, related_nodes))

        for key, value in metrics.items():
            if value is not None:
                metrics_map[key].append(value * related_concept_weight)

    # No weight normalization required as given related concepts are available.
    aggregated_metrics = {
        key: float(sum(entries)) if entries else None
        for key, entries in metrics_map.items()
    }

    return aggregated_metrics


def hypernym_paths(node: graph.AbstractNode) -> t.FrozenSet[graph.AbstractPath]:
    if kg_cn:
        return conceptnet.Database().hypernym_paths(
            t.cast(conceptnet.ConceptnetNode, node)
        )

    elif kg_wn:
        return wordnet.hypernym_paths(t.cast(wordnet.WordnetNode, node))

    return frozenset()
