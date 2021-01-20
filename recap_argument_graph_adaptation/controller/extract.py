import logging
import typing as t

import recap_argument_graph as ag
from recap_argument_graph_adaptation.model import casebase, graph, query, spacy
from recap_argument_graph_adaptation.model.config import Config

config = Config.instance()
log = logging.getLogger(__name__)

spacy_pos_tags = ["NOUN", "PROPN", "VERB", "ADJ"]  # ADV


def keywords(
    graph: ag.Graph, rules: t.Collection[casebase.Rule]
) -> t.Set[casebase.Concept]:
    related_concepts = {rule.source: 1 / len(rules) for rule in rules}
    rule_sources = {rule.source for rule in rules}
    rule_targets = {rule.target for rule in rules}

    concepts: t.Set[casebase.Concept] = set()
    mc = graph.major_claim

    keywords_response = spacy.keywords(
        [node.plain_text for node in graph.inodes], spacy_pos_tags
    )

    for node, doc in zip(graph.inodes, keywords_response):
        keywords = doc["keywords"]
        node_vector = doc["vector"]
        mc_distance = graph.node_distance(node, mc)

        for k in keywords:
            pos_tag = casebase.spacy2pos(k["pos_tag"])
            nodes = query.concept_nodes(
                k["term"], pos_tag, node_vector
            ) or query.concept_nodes(k["lemma"], pos_tag, node_vector)

            if nodes:
                candidate = casebase.Concept(
                    k["term"],
                    k["vector"],
                    pos_tag,
                    nodes,
                    query.concept_metrics(
                        related_concepts,
                        nodes,
                        k["vector"],
                        weight=k["weight"],
                        major_claim_distance=mc_distance,
                    ),
                )

                if candidate not in rule_sources and candidate not in rule_targets:
                    concepts.add(candidate)

    concepts = casebase.filter_concepts(
        concepts, config.tuning("extraction", "min_score")
    )

    log.debug(
        f"Found the following concepts: {', '.join((str(concept) for concept in concepts))}"
    )

    return concepts


def paths(
    concepts: t.Iterable[casebase.Concept], rules: t.Collection[casebase.Rule]
) -> t.Dict[casebase.Concept, t.List[graph.AbstractPath]]:
    result = {}
    method = config.tuning("bfs", "method")

    if method == "within":
        for concept in concepts:
            paths = []

            for rule in rules:
                if (
                    candidates := query.all_shortest_paths(
                        rule.source.nodes, concept.nodes
                    )
                ) is not None:
                    paths.extend(candidates)
                    log.debug(
                        f"Found {len(candidates)} reference path(s) for ({rule.source})->({concept})."
                    )

            if paths:
                result[concept] = paths

    elif method == "between":
        paths = []

        for rule in rules:
            if candidates := query.all_shortest_paths(
                rule.source.nodes, rule.target.nodes
            ):
                paths.extend(candidates)

            log.debug(
                f"Found {len(paths)} reference path(s) for ({rule.source})->({rule.target})."
            )

        if paths:
            for concept in concepts:
                result[concept] = paths

    else:
        raise ValueError("The parameter 'method' is not set correctly.")

    return result
