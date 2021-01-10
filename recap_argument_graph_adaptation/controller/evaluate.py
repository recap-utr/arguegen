import logging
import statistics
import typing as t
from dataclasses import dataclass

from recap_argument_graph_adaptation.controller import metrics, spacy
from recap_argument_graph_adaptation.helper import convert
from recap_argument_graph_adaptation.model.adaptation import Case, Concept
from recap_argument_graph_adaptation.model.evaluation import Evaluation

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeightedScore:
    score: float
    weight: float


def case(case: Case, adapted_concepts: t.Mapping[Concept, Concept]) -> Evaluation:
    case_rules = set(case.rules)
    benchmark_rules = set(case.benchmark_rules)

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
        weight = len(benchmark_adaptations) - i + 1

        if computed_adaptation := computed_adaptations.get(original_concept):
            positive_scores.append(
                WeightedScore(
                    _compute_score(benchmark_adaptation, computed_adaptation), weight
                )
            )
        else:
            negative_scores.append(WeightedScore(0.0, weight))

    for original_concept in only_computed:
        # Here, benchmark_adaptation == original_concept
        # These scores are 'penalized' due to the fact that they get a lower weight:
        # All of these concepts *combined* count as much as the least important benchmark rule.
        computed_adaptation = computed_adaptations[original_concept]
        negative_scores.append(
            WeightedScore(
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

    return Evaluation(
        global_score, benchmark_and_computed, only_benchmark, only_computed
    )


def _compute_score(
    benchmark_adaptation: Concept, computed_adaptation: Concept
) -> float:
    if benchmark_adaptation == computed_adaptation:
        return 1.0

    return spacy.similarity(benchmark_adaptation.vector, computed_adaptation.vector)

    # comparison_metrics = metrics.update_concept_metrics(
    #     computed_adaptation, benchmark_adaptation
    # )
    # comparison_concept = Concept.from_concept(computed_adaptation, comparison_metrics)

    # return comparison_concept.score
