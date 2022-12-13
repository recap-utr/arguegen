import itertools
import logging
import re
import typing as t
from collections import defaultdict

import arguebuf as ag

from arguegen.config import config, tuning
from arguegen.model import casebase, evaluation, nlp, wordnet

log = logging.getLogger(__name__)


def keywords(
    graph: ag.Graph, rules: t.Collection[casebase.Rule], user_query: casebase.UserQuery
) -> t.Tuple[t.Set[casebase.Concept], t.Set[casebase.Concept]]:
    related_concepts = {rule.source: 1 / len(rules) for rule in rules}
    rule_sources = {rule.source for rule in rules}
    rule_targets = {rule.target for rule in rules}

    candidates: set[casebase.Concept] = set()
    mc = graph.major_claim or graph.root_node
    assert mc is not None

    use_mc_proximity = "major_claim_prox" in tuning(config, "score")

    keywords = nlp.keywords(
        [node.plain_text.lower() for node in graph.atom_nodes.values()],
        tuning(config, "extraction", "keyword_pos_tags"),
    )

    for k in keywords:
        lemma = k.lemma
        form2pos = k.form2pos
        pos2form = k.pos2form
        pos = casebase.spacy2pos(k.pos_tag)
        weight = k.weight

        atoms = set()

        for form in form2pos:
            pattern = re.compile(f"\\b({form})\\b")

            for atom in graph.atom_nodes.values():
                node_txt = atom.plain_text.lower()

                if pattern.search(node_txt):
                    atoms.add(t.cast(casebase.HashableAtom, atom))

        if len(atoms) == 0:
            continue

        mc_distance = None

        if use_mc_proximity:
            mc_distances = set()

            for atom in atoms:
                if mc_distance := graph.node_distance(atom, mc):
                    mc_distances.add(mc_distance)

            if mc_distances:
                mc_distance = min(mc_distances)

        kg_nodes = wordnet.concept_synsets(
            form2pos.keys(),
            pos,
            [atom.plain_text for atom in atoms],
            tuning(config, "threshold", "node_similarity", "extraction"),
        )

        if len(kg_nodes) > 0:
            candidate = casebase.Concept(
                lemma.lower(),
                form2pos,
                pos2form,
                pos,
                frozenset(atoms),
                kg_nodes,
                related_concepts,
                user_query,
                evaluation.concept_metrics(
                    "extraction",
                    related_concepts,
                    user_query,
                    atoms,
                    kg_nodes,
                    lemma,
                    weight=weight,
                    major_claim_distance=mc_distance,
                ),
            )

            if candidate not in rule_sources and candidate not in rule_targets:
                candidates.add(candidate)

    occurences: defaultdict[casebase.Concept, int] = defaultdict(int)

    for c in candidates:
        for form in c.forms:
            pattern = re.compile(f"\\b({form})\\b", re.IGNORECASE)

            for adu in c.atoms:
                occurences[c] += len(re.findall(pattern, adu.plain_text))

    for (c1, o1), (c2, o2) in itertools.product(occurences.items(), occurences.items()):
        # 'tuition' in 'tuition fees'
        if (
            c1 != c2
            and (c2.lemma.startswith(c1.lemma) or c2.lemma.endswith(c1.lemma))
            and o1 == o2
        ):
            candidates.difference_update([c1])

    concepts = casebase.filter_concepts(
        candidates,
        tuning(config, "threshold", "concept_score", "extraction"),
        tuning(config, "extraction", "max_concepts"),
    )

    log.debug(
        f"Found the following concepts: {', '.join((str(concept) for concept in concepts))}"
    )

    return concepts, candidates


def paths(
    concepts: t.Iterable[casebase.Concept], rules: t.Collection[casebase.Rule]
) -> t.Dict[casebase.Concept, t.List[wordnet.Path]]:
    result = {}
    bfs_method = "between"

    if bfs_method == "within":
        for concept in concepts:
            paths = []

            for rule in rules:
                if (
                    candidates := wordnet.all_shortest_paths(
                        rule.source.synsets, concept.synsets
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
            if candidates := wordnet.all_shortest_paths(
                rule.source.synsets, rule.target.synsets
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
