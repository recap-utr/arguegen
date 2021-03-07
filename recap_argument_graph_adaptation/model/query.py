import itertools
import statistics
import typing as t

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
    names: t.Iterable[str],
    pos: t.Optional[casebase.POS],
    comparison_vectors: t.Optional[t.Iterable[spacy.Vector]] = None,
    min_similarity: t.Optional[float] = None,
) -> t.FrozenSet[graph.AbstractNode]:
    if kg_wn:
        return wordnet.concept_synsets(names, pos, comparison_vectors, min_similarity)

    elif kg_cn:
        return conceptnet.Database().nodes(names, pos)

    raise kg_err


def concept_metrics(
    stage: str,
    related_concepts: t.Union[casebase.Concept, t.Mapping[casebase.Concept, float]],
    user_query: casebase.UserQuery,
    inodes: t.Iterable[casebase.ArgumentNode],
    nodes: t.Iterable[graph.AbstractNode],
    vector: spacy.Vector,
    weight: t.Optional[float] = None,
    hypernym_level: t.Optional[int] = None,
    hypernym_proximity: t.Optional[float] = None,
    major_claim_distance: t.Optional[int] = None,
    major_claim_proximity: t.Optional[float] = None,
) -> t.Dict[str, t.Optional[float]]:
    if isinstance(related_concepts, casebase.Concept):
        related_concepts = {related_concepts: 1.0}

    if not (hypernym_level is None or hypernym_proximity is None) or not (
        major_claim_distance is None or major_claim_proximity is None
    ):
        raise RuntimeError("Invalid combination of metric parameters")

    assert stage in casebase.metrics_per_stage.keys()

    active_metrics = config.tuning("score").keys()
    active = lambda x: x in active_metrics and x in casebase.metrics_per_stage[stage]

    total_weight = 0
    metrics_map = {key: [] for key in casebase.metric_keys}

    for related_concept, related_concept_weight in related_concepts.items():
        total_weight += related_concept_weight
        concept_semantic_similarity = (
            spacy.similarity(vector, related_concept.vector)
            if active("concept_sem_sim")
            else None
        )
        query_concept_semantic_similarity = (
            spacy.similarity(user_query.vector, vector)
            if active("query_concept_sem_sim")
            else None
        )
        adus_semantic_similarity = (
            statistics.mean(
                spacy.similarity(inode2.vector, inode1.vector)
                for inode1, inode2 in itertools.product(related_concept.inodes, inodes)
            )
            if active("adus_sem_sim")
            else None
        )
        query_adus_semantic_similarity = (
            statistics.mean(
                spacy.similarity(user_query.vector, inode.vector) for inode in inodes
            )
            if active("query_adus_sem_sim")
            else None
        )

        metrics = {
            "adus_sem_sim": adus_semantic_similarity,
            "concept_sem_sim": concept_semantic_similarity,
            "hypernym_prox": hypernym_proximity or _dist2sim(hypernym_level),
            "keyword_weight": weight,
            "major_claim_prox": major_claim_proximity
            or _dist2sim(major_claim_distance),
            "nodes_path_sim": None,
            "nodes_sem_sim": None,
            "nodes_wup_sim": None,
            "query_adus_sem_sim": query_adus_semantic_similarity,
            "query_concept_sem_sim": query_concept_semantic_similarity,
            "query_nodes_sem_sim": None,
        }

        assert metrics.keys() == casebase.metric_keys

        if kg_wn:
            nodes = t.cast(t.Iterable[wordnet.WordnetNode], nodes)
            related_nodes = t.cast(
                t.Iterable[wordnet.WordnetNode], related_concept.nodes
            )

            metrics.update(wordnet.metrics(nodes, related_nodes, active))
            metrics["query_nodes_sem_sim"] = (
                wordnet.query_nodes_similarity(nodes, user_query)
                if active("query_nodes_sem_sim")
                else None
            )

        elif kg_cn:
            db = conceptnet.Database()
            nodes = t.cast(t.Iterable[conceptnet.ConceptnetNode], nodes)
            related_nodes = t.cast(
                t.Iterable[conceptnet.ConceptnetNode], related_concept.nodes
            )

            metrics.update(db.metrics(nodes, related_nodes, active))

        for key, value in metrics.items():
            if value is not None:
                metrics_map[key].append(value * related_concept_weight)

    # No weight normalization required as given related concepts are available.
    aggregated_metrics = {
        key: float(sum(entries) / total_weight) if entries else None
        for key, entries in metrics_map.items()
    }

    return aggregated_metrics


def direct_hypernyms(
    node: graph.AbstractNode,
    comparison_vectors: t.Iterable[spacy.Vector],
    min_similarity: float,
) -> t.FrozenSet[graph.AbstractPath]:
    if kg_cn:
        return conceptnet.Database().direct_hypernyms(
            t.cast(conceptnet.ConceptnetNode, node)
        )

    elif kg_wn:
        return wordnet.direct_hypernyms(
            t.cast(wordnet.WordnetNode, node), comparison_vectors, min_similarity
        )

    raise kg_err


# def hypernym_distances(
#     node: graph.AbstractNode,
# ) -> t.Dict[graph.AbstractNode, int]:
#     if kg_cn:
#         return t.cast(
#             t.Dict[graph.AbstractNode, int],
#             conceptnet.Database().hypernym_distances(
#                 t.cast(conceptnet.ConceptnetNode, node)
#             ),
#         )

#     elif kg_wn:
#         return t.cast(
#             t.Dict[graph.AbstractNode, int],
#             t.cast(wordnet.WordnetNode, node).hypernym_distances(),
#         )

#     raise kg_err


def all_shortest_paths(
    start_nodes: t.Iterable[graph.AbstractNode],
    end_nodes: t.Iterable[graph.AbstractNode],
) -> t.FrozenSet[graph.AbstractPath]:
    if kg_cn:
        all_paths = conceptnet.Database().all_shortest_paths(
            t.cast(t.Iterable[conceptnet.ConceptnetNode], start_nodes),
            t.cast(t.Iterable[conceptnet.ConceptnetNode], end_nodes),
        )

    elif kg_wn:
        all_paths = wordnet.all_shortest_paths(
            t.cast(t.Iterable[wordnet.WordnetNode], start_nodes),
            t.cast(t.Iterable[wordnet.WordnetNode], end_nodes),
        )

    else:
        raise kg_err

    # if all_paths:
    #     shortest_length = min(len(path) for path in all_paths)

    #     return frozenset(path for path in all_paths if len(path) == shortest_length)

    return all_paths
