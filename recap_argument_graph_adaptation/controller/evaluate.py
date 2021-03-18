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
) -> casebase.EvaluationTuple:
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

    tn = {x for x in all_concepts if x not in benchmark_keys and x not in computed_keys}

    log.debug(f"Common adaptations: {convert.list_str(benchmark_and_computed)}")
    log.debug(f"Only benchmark adaptations: {convert.list_str(only_benchmark)}")
    log.debug(f"Only computed adaptations: {convert.list_str(only_computed)}")

    tp, fn, fp = _eval_sets(
        computed_adaptations, benchmark_adaptations, benchmark_weights, case.user_query
    )
    tp_score, fn_score, fp_score = _eval_scores(
        tp,
        fn,
        fp,
        benchmark_weights,
        computed_adaptations,
    )

    retrieved_sim = None
    adapted_sim = None

    if adapted_graph:
        retrieved_sim = _graph_similarity(case.user_query, case.graph)
        adapted_sim = _graph_similarity(case.user_query, adapted_graph)

    eval_result = casebase.Evaluation(
        duration=duration,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        tp_score=tp_score,
        fn_score=fn_score,
        fp_score=fp_score,
        retrieved_sim=retrieved_sim,
        adapted_sim=adapted_sim,
    )

    log.debug(f"Finished with global score of {eval_result.score}.")

    baseline_adaptations = {concept: concept for concept in computed_adaptations}

    tp_baseline, fn_baseline, fp_baseline = _eval_sets(
        baseline_adaptations, benchmark_adaptations, benchmark_weights, case.user_query
    )
    tp_score_baseline, fn_score_baseline, fp_score_baseline = _eval_scores(
        tp_baseline,
        fn_baseline,
        fp_baseline,
        benchmark_weights,
        baseline_adaptations,
    )

    baseline_result = casebase.Evaluation(
        duration=0,
        tp=tp_baseline,
        tn=set(),
        fp=fp_baseline,
        fn=fn_baseline,
        tp_score=tp_score_baseline,
        fn_score=fn_score_baseline,
        fp_score=fp_score_baseline,
        retrieved_sim=retrieved_sim,
        adapted_sim=retrieved_sim,
    )

    return casebase.EvaluationTuple(eval_result, baseline_result)


def _eval_sets(
    computed_adaptations: t.Mapping[casebase.Concept, casebase.Concept],
    benchmark_adaptations: t.Mapping[casebase.Concept, casebase.Concept],
    benchmark_weights: t.Iterable[int],
    user_query: casebase.UserQuery,
) -> t.Tuple[
    t.List[casebase.WeightedScore],
    t.List[casebase.WeightedScore],
    t.List[casebase.WeightedScore],
]:
    tp = []
    fn = []
    fp = []
    tp_baseline = []

    for weight, (original_concept, benchmark_adaptation) in zip(
        benchmark_weights, benchmark_adaptations.items()
    ):
        if computed_adaptation := computed_adaptations.get(original_concept):
            tp.append(
                casebase.WeightedScore(
                    original_concept,
                    _compute_score(
                        benchmark_adaptation, computed_adaptation, user_query
                    ),
                    weight,
                )
            )
            tp_baseline.append(
                casebase.WeightedScore(
                    original_concept,
                    _compute_score(benchmark_adaptation, original_concept, user_query),
                    weight,
                )
            )
        else:
            fn.append(casebase.WeightedScore(original_concept, 0.0, weight))

    for original_concept, computed_adaptation in computed_adaptations.items():
        if original_concept not in benchmark_adaptations:
            # Here, benchmark_adaptation == original_concept
            fp.append(
                casebase.WeightedScore(
                    original_concept,
                    _compute_score(original_concept, computed_adaptation, user_query),
                    1,
                )
            )

    return tp, fn, fp


def _eval_scores(
    true_positives: t.Iterable[casebase.WeightedScore],
    false_negatives: t.Iterable[casebase.WeightedScore],
    false_positives: t.Iterable[casebase.WeightedScore],
    benchmark_weights: t.Iterable[int],
    computed_adaptations: t.Mapping[casebase.Concept, casebase.Concept],
) -> t.Tuple[float, float, float]:
    tp_score = 0.0
    fn_score = 0.0
    fp_score = 0.0

    tp_denominator = sum(x.weight for x in true_positives)

    if tp_denominator > 0:
        tp_score = sum(x.score * x.weight for x in true_positives) / tp_denominator

    fn_denominator = sum(benchmark_weights)

    if fn_denominator > 0:
        fn_score = sum(x.weight for x in false_negatives) / fn_denominator

    fp_denominator = sum(
        1 - _compute_score(concept, adaptation, case.user_query)
        for concept, adaptation in computed_adaptations.items()
    )

    if fp_denominator:
        fp_score = (sum(1 - x.score for x in false_positives)) / fp_denominator

    return tp_score, fn_score, fp_score


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
