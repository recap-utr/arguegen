# TODO: Wenn mehrere Regeln vorhanden sind, sollte man diese nicht als mehrere isolierte Adaptionen betrachtet.
# Stattdessen kann man alle Regeln gleichzeitig betrachten (es gibt dann einfach mehr fest definierte Adaptionen)
# und so die Adaptionen besser machen. Das ist möglich, weil es dann mehr kontextuelles Wissen gibt.

# TODO: Anstatt der semantischen Ähnlichkeit kann man den Score auch hier benutzen.

import statistics
import typing as t
from recap_argument_graph_adaptation.model.adaptation import Case, Concept


def case(case: Case, adapted_concepts: t.Mapping[Concept, Concept]) -> float:
    benchmark_adaptations = {
        original_concept: original_concept
        for original_concept in adapted_concepts.keys()
    }
    computed_adaptations = {**adapted_concepts}

    benchmark_adaptations.update(
        {rule.source: rule.target for rule in case.benchmark_rules}
    )
    computed_adaptations.update({rule.source: rule.target for rule in case.rules})

    similarities = []

    for original_concept, computed_adaptation in computed_adaptations.items():
        benchmark_adaptation = benchmark_adaptations[original_concept]

        if computed_adaptation == benchmark_adaptation:
            similarities.append(1)
        else:
            similarities.append(
                computed_adaptation.name.similarity(benchmark_adaptation.name)
            )

    return statistics.mean(similarities)
