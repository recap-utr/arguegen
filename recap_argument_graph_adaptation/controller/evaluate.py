import logging
import statistics
import typing as t

from recap_argument_graph_adaptation.controller import metrics
from recap_argument_graph_adaptation.helper import convert
from recap_argument_graph_adaptation.model.adaptation import Case, Concept
from recap_argument_graph_adaptation.model.evaluation import Evaluation

log = logging.getLogger(__name__)


def case(case: Case, adapted_concepts: t.Mapping[Concept, Concept]) -> Evaluation:
    computed_adaptations = {**adapted_concepts}
    computed_adaptations.update({rule.source: rule.target for rule in case.rules})
    benchmark_adaptations = {rule.source: rule.target for rule in case.benchmark_rules}

    benchmark_keys = set(benchmark_adaptations)
    computed_keys = set(computed_adaptations)

    benchmark_and_computed = {k for k in benchmark_keys if k in computed_keys}
    only_benchmark = {k for k in benchmark_keys - computed_keys}
    only_computed = {k for k in computed_keys - benchmark_keys}

    log.debug(f"Common adaptations: {convert.list_str(benchmark_and_computed)}")
    log.debug(f"Only benchmark adaptations: {convert.list_str(only_benchmark)}")
    log.debug(f"Only computed adaptations: {convert.list_str(only_computed)}")

    scores = []

    for original_concept in benchmark_and_computed:
        benchmark_adaptation = benchmark_adaptations[original_concept]
        computed_adaptation = computed_adaptations[original_concept]
        scores.append(_compute_score(benchmark_adaptation, computed_adaptation))

    # Here, a penalty is applied, because we assume that ignoring a specified adaptation is the worst case.
    for original_concept in only_benchmark:
        benchmark_adaptation = benchmark_adaptations[original_concept]
        # scores.append(0.5 * _compute_score(benchmark_adaptation, original_concept))
        scores.append(0.0)

    for original_concept in only_computed:
        computed_adaptation = computed_adaptations[original_concept]
        scores.append(0.5 * _compute_score(original_concept, computed_adaptation))

    mean = statistics.mean(scores)
    log.debug(f"Finished with global score of {round(mean, 3)}.")

    return Evaluation(mean, benchmark_and_computed, only_benchmark, only_computed)


def _compute_score(
    benchmark_adaptation: Concept, computed_adaptation: Concept
) -> float:
    if benchmark_adaptation == computed_adaptation:
        return 1.0

    comparison_metrics = metrics.update_concept_metrics(
        computed_adaptation, benchmark_adaptation
    )
    comparison_concept = Concept.from_concept(computed_adaptation, comparison_metrics)

    return comparison_concept.score
