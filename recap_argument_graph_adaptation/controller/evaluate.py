import logging
import typing as t
from dataclasses import dataclass

from recap_argument_graph_adaptation.controller import convert
from recap_argument_graph_adaptation.model import casebase, query, spacy

log = logging.getLogger(__name__)


def case(
    case: casebase.Case, adapted_concepts: t.Mapping[casebase.Concept, casebase.Concept]
) -> casebase.Evaluation:
    case_rules = case.rules
    benchmark_rules = case.benchmark_rules

    computed_adaptations = {**adapted_concepts}
    benchmark_adaptations = {
        rule.source: rule.target for rule in benchmark_rules if rule not in case_rules
    }

    benchmark_keys = set(benchmark_adaptations)
    computed_keys = set(computed_adaptations)

    benchmark_and_computed = {k for k in benchmark_keys if k in computed_keys}
    only_benchmark = {k for k in benchmark_keys - computed_keys}
    only_computed = {k for k in computed_keys - benchmark_keys}

    log.debug(f"Common adaptations: {convert.list_str(benchmark_and_computed)}")
    log.debug(f"Only benchmark adaptations: {convert.list_str(only_benchmark)}")
    log.debug(f"Only computed adaptations: {convert.list_str(only_computed)}")

    positive_scores = []
    negative_scores = []

    for i, (original_concept, benchmark_adaptation) in enumerate(
        benchmark_adaptations.items()
    ):
        weight = len(benchmark_adaptations) - i

        if computed_adaptation := computed_adaptations.get(original_concept):
            positive_scores.append(
                casebase.WeightedScore(
                    original_concept,
                    _compute_score(benchmark_adaptation, computed_adaptation),
                    weight,
                )
            )
        else:
            negative_scores.append(
                casebase.WeightedScore(original_concept, 0.0, weight)
            )

    for original_concept in only_computed:
        # Here, benchmark_adaptation == original_concept
        # These scores are 'penalized' due to the fact that they get a lower weight.
        computed_adaptation = computed_adaptations[original_concept]
        negative_scores.append(
            casebase.WeightedScore(
                original_concept,
                _compute_score(original_concept, computed_adaptation),
                1 / len(benchmark_rules),
            )
        )

    positive_score = 0
    negative_score = 0

    if positive_scores:
        positive_score = sum(
            item.score * item.weight for item in positive_scores
        ) / sum(item.weight for item in positive_scores)

    if negative_scores:
        negative_score = sum(
            (1 - item.score) * item.weight for item in negative_scores
        ) / sum(item.weight for item in negative_scores)

    global_score = positive_score - negative_score

    log.debug(f"Finished with global score of {global_score}.")

    return casebase.Evaluation(
        global_score,
        positive_scores,
        negative_scores,
        benchmark_and_computed,
        only_benchmark,
        only_computed,
    )


def _compute_score(concept1: casebase.Concept, concept2: casebase.Concept) -> float:
    if concept1 == concept2:
        return 1.0

    # return spacy.similarity(benchmark_adaptation.vector, computed_adaptation.vector)

    scores = []

    for c1, c2 in ((concept1, concept2), (concept2, concept1)):
        metrics = query.concept_metrics(
            c2,
            c1.nodes,
            c1.vector,
        )
        scores.append(casebase.score(metrics))

    return max(scores)
