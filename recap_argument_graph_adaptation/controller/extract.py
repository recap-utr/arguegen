import itertools
import logging
import re
import typing as t
from collections import defaultdict

import arguebuf as ag
from recap_argument_graph_adaptation.model import casebase, graph, query, nlp
from recap_argument_graph_adaptation.model.config import Config

config = Config.instance()
log = logging.getLogger(__name__)


def keywords(
    graph: ag.Graph, rules: t.Collection[casebase.Rule], user_query: casebase.UserQuery
) -> t.Tuple[t.Set[casebase.Concept], t.Set[casebase.Concept]]:
    related_concepts = {rule.source: 1 / len(rules) for rule in rules}
    rule_sources = {rule.source for rule in rules}
    rule_targets = {rule.target for rule in rules}

    candidates = set()
    mc = graph.major_claim
    use_mc_proximity = "major_claim_prox" in config.tuning("score")

    keywords = nlp.keywords(
        [node.plain_text.lower() for node in graph.atom_nodes.values()],
        config.tuning("extraction", "keyword_pos_tags"),
    )

    for k in keywords:
        kw = k.keyword
        kw_form2pos = k.form2pos
        kw_pos2form = k.pos2form
        kw_pos = casebase.spacy2pos(k.pos_tag)
        kw_weight = k.weight

        inodes = set()

        for kw_form in kw_form2pos:
            pattern = re.compile(f"\\b({kw_form})\\b")

            for inode in graph.atom_nodes.values():
                node_txt = inode.plain_text.lower()

                if pattern.search(node_txt):
                    inodes.add(t.cast(casebase.HashableAtom, inode))

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
            kw_form2pos.keys(),
            kw_pos,
            [inode.plain_text for inode in inodes],
            config.tuning("threshold", "node_similarity", "extraction"),
        )

        if len(kg_nodes) > 0:
            candidate = casebase.Concept(
                kw.lower(),
                kw_form2pos,
                kw_pos2form,
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
                    kw,
                    weight=kw_weight,
                    major_claim_distance=mc_distance,
                ),
            )

            if candidate not in rule_sources and candidate not in rule_targets:
                candidates.add(candidate)

    occurences = defaultdict(int)

    for c in candidates:
        for form in c.forms:
            pattern = re.compile(f"\\b({form})\\b", re.IGNORECASE)

            for adu in c.inodes:
                occurences[c] += len(re.findall(pattern, adu.plain_text))

    for (c1, o1), (c2, o2) in itertools.product(occurences.items(), occurences.items()):
        # 'tuition' in 'tuition fees'
        if (
            c1 != c2
            and (c2.name.startswith(c1.name) or c2.name.endswith(c1.name))
            and o1 == o2
        ):
            candidates.difference_update([c1])

    concepts = casebase.filter_concepts(
        candidates,
        config.tuning("threshold", "concept_score", "extraction"),
        config.tuning("extraction", "max_concepts"),
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
