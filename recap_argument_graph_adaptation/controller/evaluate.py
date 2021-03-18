import logging
import typing as t

import recap_argument_graph as ag
from recap_argument_graph_adaptation.controller import convert
from recap_argument_graph_adaptation.model import casebase, query, spacy

log = logging.getLogger(__name__)


def case(
    case: casebase.Case,
    adapted_concepts: t.Mapping[casebase.Concept, casebase.Concept],
    all_concepts: t.Iterable[casebase.Concept],
    adapted_graph: t.Optional[ag.Graph],
    duration: float,
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
    only_computed = {k for k in computed_keys - benchmark_keys}

    log.debug(f"Common adaptations: {convert.list_str(benchmark_and_computed)}")
    log.debug(f"Only benchmark adaptations: {convert.list_str(only_benchmark)}")
    log.debug(f"Only computed adaptations: {convert.list_str(only_computed)}")

    true_positives = []
    true_positives_benchmark = []
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
            true_positives_benchmark.append(
                casebase.WeightedScore(
                    original_concept,
                    _compute_score(
                        benchmark_adaptation, original_concept, case.user_query
                    ),
                    weight,
                )
            )
        else:
            false_negatives.append(
                casebase.WeightedScore(original_concept, 0.0, weight)
            )

    for original_concept, computed_adaptation in computed_adaptations.items():
        if original_concept not in benchmark_adaptations:
            # Here, benchmark_adaptation == original_concept
            false_positives.append(
                casebase.WeightedScore(
                    original_concept,
                    _compute_score(
                        original_concept, computed_adaptation, case.user_query
                    ),
                    fp_weight,
                )
            )

    tp_score = _tp_score(true_positives)
    benchmark_tp_score = _tp_score(true_positives_benchmark)
    fn_score = 0.0
    fp_score = 0.0

    fn_denominator = sum(benchmark_weights)

    if fn_denominator > 0:
        fn_score = sum(x.weight for x in false_negatives) / fn_denominator

    fp_denominator = sum(
        1 - _compute_score(concept, adaptation, case.user_query)
        for concept, adaptation in computed_adaptations.items()
    )

    if fp_denominator:
        fp_score = (sum(1 - x.score for x in false_positives)) / fp_denominator

    retrieved_sim = None
    adapted_sim = None

    if adapted_graph:
        retrieved_sim = _graph_similarity(case.user_query, case.graph)
        adapted_sim = _graph_similarity(case.user_query, adapted_graph)

    eval_result = casebase.Evaluation(
        duration=duration,
        tp=true_positives,
        tn=true_negatives,
        fp=false_positives,
        fn=false_negatives,
        tp_score=tp_score,
        fn_score=fn_score,
        fp_score=fp_score,
        retrieved_sim=retrieved_sim,
        adapted_sim=adapted_sim,
        benchmark_tp_score=benchmark_tp_score,
    )

    log.debug(f"Finished with global score of {eval_result.score}.")

    return eval_result


def _tp_score(true_positives: t.Iterable[casebase.WeightedScore]) -> float:
    tp_denominator = sum(x.weight for x in true_positives)

    if tp_denominator > 0:
        return sum(x.score * x.weight for x in true_positives) / tp_denominator

    return 0.0


def _graph_similarity(user_query: casebase.UserQuery, graph: ag.Graph) -> float:
    graph_text = " ".join(inode.plain_text for inode in graph.inodes)

    return spacy.similarity(user_query.vector, graph_text)


def _compute_score(
    concept1: casebase.Concept,
    concept2: casebase.Concept,
    user_query: casebase.UserQuery,
) -> float:
    if concept1 == concept2:
        return 1.0

    return casebase.score(
        query.concept_metrics(
            "evaluation",
            concept2,
            user_query,
            concept1.inodes,
            concept1.nodes,
            concept1.vector,
        )
    )
