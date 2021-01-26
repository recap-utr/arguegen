import itertools
import logging
import typing as t
from collections import defaultdict
from dataclasses import dataclass

from recap_argument_graph_adaptation.controller import convert
from recap_argument_graph_adaptation.model import casebase, query, spacy

log = logging.getLogger(__name__)


def case(
    case: casebase.Case,
    adapted_concepts: t.Mapping[casebase.Concept, casebase.Concept],
    all_concepts: t.List[casebase.Concept],
) -> casebase.Evaluation:
    case_rules = case.rules
    benchmark_rules = case.benchmark_rules

    benchmark_adaptations = {
        rule.source: rule.target for rule in benchmark_rules if rule not in case_rules
    }
    computed_adaptations = {**adapted_concepts}
    benchmark_weights = list(range(len(benchmark_adaptations), 0, -1))

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

    true_positives = []
    false_negatives = []
    false_positives = []
    fp_weight = 1 / len(benchmark_rules)

    true_negatives = {
        x for x in all_concepts if x not in benchmark_keys and x not in computed_keys
    }

    for weight, (original_concept, benchmark_adaptation) in zip(
        benchmark_weights, benchmark_adaptations.items()
    ):
        if computed_adaptation := computed_adaptations.get(original_concept):
            true_positives.append(
                casebase.WeightedScore(
                    original_concept,
                    _compute_score(
                        benchmark_adaptation, computed_adaptation, case.user_query
                    ),
                    weight,
                )
            )
        else:
            false_negatives.append(
                casebase.WeightedScore(original_concept, 0.0, weight)
            )

    for original_concept in only_computed:
        # Here, benchmark_adaptation == original_concept
        computed_adaptation = computed_adaptations[original_concept]
        false_positives.append(
            casebase.WeightedScore(
                original_concept,
                _compute_score(original_concept, computed_adaptation, case.user_query),
                fp_weight,
            )
        )

    tp_score = 0.0
    fn_score = 0.0
    fp_score = 0.0

    if true_positives:
        tp_score = sum(x.score * x.weight for x in true_positives) / sum(
            x.weight for x in true_positives
        )

    if false_negatives:
        fn_score = sum(x.weight for x in false_negatives) / sum(benchmark_weights)

    if false_positives:
        fp_score = (
            # fp_weight *
            (sum(1 - x.score for x in false_positives))
            / (
                sum(
                    1 - _compute_score(concept, adaptation, case.user_query)
                    for concept, adaptation in computed_adaptations.items()
                )
            )
        )

    eval_result = casebase.Evaluation(
        tp=true_positives,
        tn=true_negatives,
        fp=false_positives,
        fn=false_negatives,
        tp_score=tp_score,
        fn_score=fn_score,
        fp_score=fp_score,
    )

    log.debug(f"Finished with global score of {eval_result.score}.")

    return eval_result


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
