import itertools
import logging
import re
import typing as t
from collections import defaultdict

from arguegen.config import BfsMethod, ExtractionConfig, ScoreConfig
from arguegen.controllers import scorer
from arguegen.model import casebase, wordnet
from arguegen.model.nlp import Nlp

log = logging.getLogger(__name__)


def keywords(
    case: casebase.Case,
    nlp: Nlp,
    config: ExtractionConfig,
    score_config: ScoreConfig,
    wn: wordnet.Wordnet,
) -> t.Tuple[set[casebase.ScoredConcept], set[casebase.ScoredConcept]]:
    related_concepts = {rule.source: 1 / len(case.rules) for rule in case.rules}
    rule_sources = {rule.source for rule in case.rules}
    rule_targets = {rule.target for rule in case.rules}

    candidates: set[casebase.ScoredConcept] = set()

    keywords = nlp.keywords(
        [node.plain_text.lower() for node in case.case_graph.atom_nodes.values()],
        config.keyword_pos_tags,
        config.keywords_per_adu,
    )

    for k in keywords:
        lemma = k.lemma
        form2pos = k.form2pos
        pos2form = k.pos2form
        pos = casebase.spacy2pos(k.pos_tag)

        atoms = set()

        for form in form2pos:
            pattern = re.compile(f"\\b({form})\\b")

            for atom in case.case_graph.atom_nodes.values():
                node_txt = atom.plain_text.lower()

                if pattern.search(node_txt):
                    atoms.add(atom)

        if len(atoms) == 0:
            continue

        kg_nodes = wn.concept_synsets(
            form2pos.keys(),
            pos,
            nlp,
            [atom.plain_text for atom in atoms],
            config.synset_similarity_threshold,
        )

        if len(kg_nodes) > 0:
            candidate = casebase.Concept(
                lemma.lower(),
                form2pos,
                pos2form,
                pos,
                frozenset(atoms),
                kg_nodes,
            )

            if candidate not in rule_sources and candidate not in rule_targets:
                score = scorer.Scorer(
                    candidate,
                    case.case_graph,
                    case.query_graph,
                    tuple(related_concepts.items()),
                    score_config,
                    nlp,
                    k.weight,
                ).compute()
                candidates.add(casebase.ScoredConcept(candidate, score))

    occurences: defaultdict[casebase.ScoredConcept, int] = defaultdict(int)

    for entry in candidates:
        c = entry.concept

        for form in c.forms:
            pattern = re.compile(f"\\b({form})\\b", re.IGNORECASE)

            for adu in c.atoms:
                occurences[entry] += len(re.findall(pattern, adu.plain_text))

    for (c1, o1), (c2, o2) in itertools.product(occurences.items(), occurences.items()):
        # 'tuition' in 'tuition fees'
        if (
            c1 != c2
            and (
                c2.concept.lemma.startswith(c1.concept.lemma)
                or c2.concept.lemma.endswith(c1.concept.lemma)
            )
            and o1 == o2
        ):
            candidates.remove(c1)

    concepts = scorer.filter_concepts(
        candidates,
        config.concept_score_threshold,
        config.concept_limit,
    )

    log.debug(
        "Found the following concepts:"
        f" {', '.join((str(concept) for concept in concepts))}"
    )

    return concepts, {entry for entry in candidates if entry not in concepts}


def paths(
    sources: t.Iterable[casebase.ScoredConcept],
    rules: t.Collection[casebase.Rule[casebase.Concept]],
    bfs_method: BfsMethod,
) -> t.Dict[casebase.ScoredConcept, t.List[wordnet.Path]]:
    result = {}

    if bfs_method == BfsMethod.WITHIN:
        for source in sources:
            paths = []

            for rule in rules:
                if (
                    candidates := wordnet.all_shortest_paths(
                        rule.source.synsets, source.concept.synsets
                    )
                ) is not None:
                    paths.extend(candidates)
                    log.debug(
                        f"Found {len(candidates)} reference path(s) for"
                        f" ({rule.source})->({source.concept})."
                    )

            if paths:
                result[source] = paths

    elif bfs_method == BfsMethod.BETWEEN:
        paths = []

        for rule in rules:
            if candidates := wordnet.all_shortest_paths(
                rule.source.synsets, rule.target.synsets
            ):
                paths.extend(candidates)

            log.debug(
                f"Found {len(paths)} reference path(s) for"
                f" ({rule.source})->({rule.target})."
            )

        if paths:
            for source in sources:
                result[source] = paths

    else:
        raise ValueError("The parameter 'bfs_method' is not set correctly.")

    return result
