# TODO: Wenn mehrere Regeln vorhanden sind, sollte man diese nicht als mehrere isolierte Adaptionen betrachtet.
# Stattdessen kann man alle Regeln gleichzeitig betrachten (es gibt dann einfach mehr fest definierte Adaptionen)
# und so die Adaptionen besser machen. Das ist möglich, weil es dann mehr kontextuelles Wissen gibt.

# TODO: Eine Art Gridsearch für Wordnet erstellen, bei der nach dem Score optimiert wird.

import logging
import statistics
import typing as t
from recap_argument_graph_adaptation.model.adaptation import Case, Concept
from recap_argument_graph_adaptation.controller import metrics
from recap_argument_graph_adaptation.controller import export

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
log = logging.getLogger(__name__)


# def case(case: Case, adapted_concepts: t.Mapping[Concept, Concept]) -> float:
#     benchmark_adaptations = {
#         original_concept: original_concept
#         for original_concept in adapted_concepts.keys()
#     }
#     computed_adaptations = {**adapted_concepts}

#     benchmark_adaptations.update(
#         {rule.source: rule.target for rule in case.benchmark_rules}
#     )
#     computed_adaptations.update({rule.source: rule.target for rule in case.rules})

#     similarities = []

#     for original_concept, computed_adaptation in computed_adaptations.items():
#         benchmark_adaptation = benchmark_adaptations[original_concept]

#         if computed_adaptation == benchmark_adaptation:
#             similarities.append(1)
#         else:
#             similarities.append(
#                 computed_adaptation.name.similarity(benchmark_adaptation.name)
#             )

#     mean = statistics.mean(similarities)

#     log.info(f"Finished with global score of {round(mean, 3)}.")


def case(case: Case, adapted_concepts: t.Mapping[Concept, Concept]) -> float:
    computed_adaptations = {**adapted_concepts}
    computed_adaptations.update({rule.source: rule.target for rule in case.rules})
    benchmark_adaptations = {rule.source: rule.target for rule in case.benchmark_rules}

    benchmark_keys = set(benchmark_adaptations)
    computed_keys = set(computed_adaptations)

    benchmark_and_computed = {
        k: benchmark_adaptations[k] for k in benchmark_keys if k in computed_keys
    }
    only_benchmark = {
        k: benchmark_adaptations[k] for k in benchmark_keys - computed_keys
    }
    only_computed = {k: computed_adaptations[k] for k in computed_keys - benchmark_keys}

    log.info(f"Common adaptations: {export.list_str(benchmark_and_computed.keys())}")
    log.info(f"Only benchmark adaptations: {export.list_str(only_benchmark.keys())}")
    log.info(f"Only computed adaptations: {export.list_str(only_computed.keys())}")

    scores = []

    for original_concept, benchmark_adaptation in benchmark_and_computed.items():
        computed_adaptation = computed_adaptations[original_concept]
        scores.append(_compute_score(benchmark_adaptation, computed_adaptation))

    # Here, a penalty is applied, because we assume that ignoring a specified adaptation is the worst case.
    for original_concept, benchmark_adaptation in only_benchmark.items():
        scores.append(0.5 * _compute_score(benchmark_adaptation, original_concept))

    for original_concept, computed_adaptation in only_computed.items():
        scores.append(_compute_score(original_concept, computed_adaptation))

    mean = statistics.mean(scores)
    log.info(f"Finished with global score of {round(mean, 3)}.")

    return mean


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
