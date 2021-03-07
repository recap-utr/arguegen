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
    use_mc_proximity = "major_claim_prox" in config.tuning("score")

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

        found_forms = set()
        inodes = set()

        for kw_form in kw_forms:
            pattern = re.compile(f"\\b({kw_form})\\b")

            for inode in graph.inodes:
                node_txt = inode.plain_text.lower()

                if pattern.search(node_txt):
                    inodes.add(t.cast(casebase.ArgumentNode, inode))
                    found_forms.add(kw_form)

        if len(inodes) == 0:
            continue

        mc_distance = None

        if use_mc_proximity:
            mc_distances = set()

            for inode in inodes:
                if mc_distance := graph.node_distance(inode, mc):
                    mc_distances.add(mc_distance)

            if mc_distances:
                mc_distance = min(mc_distances)

        kg_nodes = query.concept_nodes(
            kw_forms,
            kw_pos,
            [inode.vector for inode in inodes],
            config.tuning("threshold", "nodes_similarity", "extraction"),
        )

        if len(kg_nodes) > 0:
            candidate = casebase.Concept(
                kw.lower(),
                kw_vector,
                frozenset(found_forms),
                kw_pos,
                frozenset(inodes),
                kg_nodes,
                related_concepts,
                user_query,
                query.concept_metrics(
                    "extraction",
                    related_concepts,
                    user_query,
                    inodes,
                    kg_nodes,
                    kw_vector,
                    weight=kw_weight,
                    major_claim_distance=mc_distance,
                ),
            )

            if candidate not in rule_sources and candidate not in rule_targets:
                candidates.append(candidate)

    concepts = casebase.filter_concepts(
        candidates,
        config.tuning("threshold", "concept_score", "extraction"),
        config.tuning("extraction", "max_keywords"),
    )

    log.debug(
        f"Found the following concepts: {', '.join((str(concept) for concept in concepts))}"
    )

    return concepts, candidates


def paths(
    concepts: t.Iterable[casebase.Concept], rules: t.Collection[casebase.Rule]
) -> t.Dict[casebase.Concept, t.List[graph.AbstractPath]]:
    result = {}
    bfs_method = "between"

    if bfs_method == "within":
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

    elif bfs_method == "between":
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
        raise ValueError("The parameter 'bfs_method' is not set correctly.")

    return result
