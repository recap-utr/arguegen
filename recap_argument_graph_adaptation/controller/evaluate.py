import itertools
import logging
import typing as t
from collections import defaultdict
from dataclasses import dataclass

from recap_argument_graph_adaptation.controller import convert
from recap_argument_graph_adaptation.model import casebase, query, spacy

log = logging.getLogger(__name__)


def case(
    case: casebase.Case, adapted_concepts: t.Mapping[casebase.Concept, casebase.Concept]
) -> casebase.Evaluation:
    case_rules = case.rules
    benchmark_rules = case.benchmark_rules

    benchmark_adaptations = {
        rule.source: rule.target for rule in benchmark_rules if rule not in case_rules
    }
    computed_adaptations = {**adapted_concepts}

    benchmark_keys = set(benchmark_adaptations)
    computed_keys = set(computed_adaptations)

    benchmark_and_computed = {k for k in benchmark_keys if k in computed_keys}
    only_benchmark = {k for k in benchmark_keys - computed_keys}
    # only_computed = {k for k in computed_keys - benchmark_keys}

    # If 'tuition fees' and 'fees' are computed, but only 'tuition fees' is in the benchmark adaptations,
    # we should not decrease the score because of this additional adaptation.
    # This exclusion is only problematic, if a concept like 'fees' is left untouched deliberately.
    # I will ignore this case for now.
    # To improve, one could during the extraction check if the subset concept 'fees' only occurs
    # as a part of the superset 'tuition fees'. If this is the case, remove it from the candidates.
    # Otherwise, add it and evaluate it like any other concept (i.e., get rid of the following workaround).
    only_computed = {
        k
        for k in computed_keys - benchmark_keys
        if not any(
            k.name in other.name and k.part_eq(other) for other in benchmark_keys
        )
    }

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
                    _compute_score(
                        benchmark_adaptation, computed_adaptation, case.user_query
                    ),
                    weight,
                )
            )
        else:
            negative_scores.append(
                casebase.WeightedScore(original_concept, 0.0, weight)
            )

    for original_concept in only_computed:
        # Here, benchmark_adaptation == original_concept
        computed_adaptation = computed_adaptations[original_concept]
        negative_scores.append(
            casebase.WeightedScore(
                original_concept,
                _compute_score(original_concept, computed_adaptation, case.user_query),
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


def _compute_score(
    concept1: casebase.Concept,
    concept2: casebase.Concept,
    user_query: casebase.UserQuery,
) -> float:
    if concept1 == concept2:
        return 1.0

    return casebase.score(
        query.concept_metrics(concept2, user_query, concept1.nodes, concept1.vector)
    )

    # ---

    # return spacy.similarity(concept1.vector, concept2.vector)

    # ---

    # scores = []

    # for c1, c2 in ((concept1, concept2), (concept2, concept1)):
    #     metrics = query.concept_metrics(
    #         c2,
    #         c1.nodes,
    #         c1.vector,
    #     )
    #     scores.append(casebase.score(metrics))

    # return max(scores)
