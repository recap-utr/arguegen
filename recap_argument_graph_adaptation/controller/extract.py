import logging
import typing as t

import recap_argument_graph as ag
from recap_argument_graph_adaptation.model import casebase, graph, query, spacy
from recap_argument_graph_adaptation.model.config import Config

config = Config.instance()
log = logging.getLogger(__name__)

spacy_pos_tags = ["NOUN", "PROPN", "VERB", "ADJ"]  # ADV


def keywords(
    graph: ag.Graph, rules: t.Collection[casebase.Rule], user_query: casebase.UserQuery
) -> t.Set[casebase.Concept]:
    related_concepts = {rule.source: 1 / len(rules) for rule in rules}
    rule_sources = {rule.source for rule in rules}
    rule_targets = {rule.target for rule in rules}

    concepts: t.Set[casebase.Concept] = set()
    mc = graph.major_claim

    keywords = spacy.keywords(
        [node.plain_text for node in graph.inodes], spacy_pos_tags
    )

    for k in keywords:
        term = k["term"]
        term_vector = k["vector"]
        term_pos = casebase.spacy2pos(k["pos_tag"])
        term_weight = k["weight"]
        lemma = k["lemma"]

        inodes = [
            t.cast(casebase.ArgumentNode, inode)
            for inode in graph.inodes
            if term in inode.plain_text
        ]
        mc_distances = set()
        mc_distance = None

        for inode in inodes:
            if mc_distance := graph.node_distance(inode, mc):
                mc_distances.add(mc_distance)

        if mc_distances:
            mc_distance = min(mc_distances)

        # TODO: Eventually add user_query.vector
        concept_query_args = (
            term_pos,
            [inode.vector for inode in inodes],
            config.tuning("extraction", "min_synset_similarity"),
        )
        nodes = query.concept_nodes(term, *concept_query_args,) or query.concept_nodes(
            lemma,
            *concept_query_args,
        )

        if nodes:
            candidate = casebase.Concept(
                term,
                term_vector,
                term_pos,
                frozenset(inodes),
                nodes,
                query.concept_metrics(
                    related_concepts,
                    nodes,
                    term_vector,
                    weight=term_weight,
                    major_claim_distance=mc_distance,
                ),
            )

            if candidate not in rule_sources and candidate not in rule_targets:
                concepts.add(candidate)

    concepts = casebase.filter_concepts(
        concepts, config.tuning("extraction", "min_concept_score")
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
