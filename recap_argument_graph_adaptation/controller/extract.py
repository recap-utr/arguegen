import logging
import re
import typing as t

import recap_argument_graph as ag
from recap_argument_graph_adaptation.model import casebase, graph, query, spacy
from recap_argument_graph_adaptation.model.config import Config

config = Config.instance()
log = logging.getLogger(__name__)


def keywords(
    graph: ag.Graph, rules: t.Collection[casebase.Rule], user_query: casebase.UserQuery
) -> t.Tuple[t.Set[casebase.Concept], t.List[casebase.Concept]]:
    related_concepts = {rule.source: 1 / len(rules) for rule in rules}
    rule_sources = {rule.source for rule in rules}
    rule_targets = {rule.target for rule in rules}

    candidates: t.List[casebase.Concept] = []
    mc = graph.major_claim
    use_mc_proximity = "major_claim_proximity" in config.tuning("score")

    keywords = spacy.keywords(
        [node.plain_text.lower() for node in graph.inodes],
        config.tuning("extraction", "keyword_pos_tags"),
    )

    for k in keywords:
        kw = k.keyword
        kw_forms = k.forms
        kw_vector = k.vector
        kw_pos = casebase.spacy2pos(k.pos_tag)
        kw_weight = k.weight

        # TODO: Move from server to here
        inodes = [
            t.cast(casebase.ArgumentNode, inode)
            for inode in graph.inodes
            if term in inode.plain_text.lower()
        ]
        mc_distance = None

        if use_mc_proximity:
            mc_distances = set()

            for inode in inodes:
                if mc_distance := graph.node_distance(inode, mc):
                    mc_distances.add(mc_distance)

            if mc_distances:
                mc_distance = min(mc_distances)

        nodes = query.concept_nodes(
            kw_forms,
            kw_pos,
            [inode.vector for inode in inodes],
            config.tuning("extraction", "min_synset_similarity"),
        )

        if len(nodes) > 0:
            candidate = casebase.Concept(
                kw.lower(),
                kw_vector,
                frozenset([kw.lower()]),
                kw_pos,
                frozenset(inodes),
                nodes,
                query.concept_metrics(
                    related_concepts,
                    user_query,
                    nodes,
                    kw_vector,
                    weight=kw_weight,
                    major_claim_distance=mc_distance,
                ),
            )

            if candidate not in rule_sources and candidate not in rule_targets:
                candidates.append(candidate)

    topn = config.tuning("extraction", "max_keywords")

    if isinstance(topn, float):
        if not 0.0 < topn <= 1.0:
            raise ValueError(
                f"topn = {topn} is invalid; "
                "must be an int, or a float between 0.0 and 1.0"
            )

        topn = int(round(len(candidates) * topn))

    concepts = casebase.filter_concepts(
        candidates[:topn], config.tuning("extraction", "min_concept_score")
    )

    log.debug(
        f"Found the following concepts: {', '.join((str(concept) for concept in concepts))}"
    )

    return concepts, candidates


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
