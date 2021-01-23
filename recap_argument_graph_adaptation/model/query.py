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
kg_err = RuntimeError("The specified knowledge graph is not available.")


def _dist2sim(distance: t.Optional[float]) -> t.Optional[float]:
    if distance is not None:
        return 1 / (1 + distance)

    return None


def pos(tag: t.Optional[str]) -> t.Optional[casebase.POS]:
    if kg_wn:
        return casebase.wn2pos(tag)

    elif kg_cn:
        return casebase.cn2pos(tag)

    raise kg_err


def concept_nodes(
    name: str,
    pos: t.Optional[casebase.POS],
    comparison_vectors: t.Optional[t.Iterable[np.ndarray]] = None,
    min_similarity: t.Optional[float] = None,
) -> t.FrozenSet[graph.AbstractNode]:
    if kg_wn:
        return wordnet.concept_synsets(name, pos, comparison_vectors, min_similarity)

    elif kg_cn:
        return conceptnet.Database().nodes(name, pos)

    raise kg_err


def concept_metrics(
    related_concepts: t.Union[casebase.Concept, t.Mapping[casebase.Concept, float]],
    user_query: casebase.UserQuery,
    nodes: t.Iterable[graph.AbstractNode],
    vector: np.ndarray,
    weight: t.Optional[float] = None,
    hypernym_level: t.Optional[int] = None,
    major_claim_distance: t.Optional[int] = None,
) -> t.Dict[str, t.Optional[float]]:
    if isinstance(related_concepts, casebase.Concept):
        related_concepts = {related_concepts: 1.0}

    total_weight = 0
    metrics_map = {key: [] for key in casebase.metric_keys}

    for related_concept, related_concept_weight in related_concepts.items():
        total_weight += related_concept_weight
        metrics = {
            "keyword_weight": weight,
            "nodes_semantic_similarity": None,
            "concept_semantic_similarity": spacy.similarity(
                vector, related_concept.vector
            ),
            "hypernym_proximity": _dist2sim(hypernym_level),
            "major_claim_proximity": _dist2sim(major_claim_distance),
            "nodes_path_similarity": None,
            "nodes_wup_similarity": None,
            "query_nodes_semantic_similarity": None,
            "query_concept_semantic_similarity": spacy.similarity(
                user_query.vector, vector
            ),
        }

        assert metrics.keys() == casebase.metric_keys

        if kg_wn:
            nodes = t.cast(t.Iterable[wordnet.WordnetNode], nodes)
            related_nodes = t.cast(
                t.Iterable[wordnet.WordnetNode], related_concept.nodes
            )
            metrics.update(wordnet.metrics(nodes, related_nodes))
            metrics["query_nodes_semantic_similarity"] = wordnet.query_nodes_similarity(
                nodes, user_query
            )

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
        key: float(sum(entries) / total_weight) if entries else None
        for key, entries in metrics_map.items()
    }

    return aggregated_metrics


def hypernyms_as_paths(
    node: graph.AbstractNode,
    comparison_vectors: t.Iterable[np.ndarray],
    min_similarity: float,
) -> t.FrozenSet[graph.AbstractPath]:
    if kg_cn:
        return conceptnet.Database().hypernyms_as_paths(
            t.cast(conceptnet.ConceptnetNode, node)
        )

    elif kg_wn:
        return wordnet.hypernyms_as_paths(
            t.cast(wordnet.WordnetNode, node), comparison_vectors, min_similarity
        )

    raise kg_err


def all_shortest_paths(
    start_nodes: t.Iterable[graph.AbstractNode],
    end_nodes: t.Iterable[graph.AbstractNode],
) -> t.FrozenSet[graph.AbstractPath]:
    if kg_cn:
        return conceptnet.Database().all_shortest_paths(
            t.cast(t.Iterable[conceptnet.ConceptnetNode], start_nodes),
            t.cast(t.Iterable[conceptnet.ConceptnetNode], end_nodes),
        )

    if kg_wn:
        return wordnet.all_shortest_paths(
            t.cast(t.Iterable[wordnet.WordnetNode], start_nodes),
            t.cast(t.Iterable[wordnet.WordnetNode], end_nodes),
        )

    raise kg_err
