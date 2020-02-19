import typing as t

from recap_argument_graph_adaptation.model import graph
from recap_argument_graph_adaptation.model.adaptation import Concept


def statistic(
    concepts: t.Iterable[Concept],
    reference_paths: t.Mapping[Concept, t.Iterable[graph.Path]],
    adapted_concepts: t.Mapping[Concept, str],
    adapted_paths: t.Mapping[Concept, t.Iterable[graph.Path]],
) -> t.Dict[str, t.Any]:
    out = {}

    for concept in concepts:
        out[concept.original_name] = {
            "concept": concept.conceptnet_name,
            "adaptation": adapted_concepts.get(concept),
            "reference_paths": reference_paths.get(concept),
            "adapted_paths": adapted_paths.get(concept),
        }

    return out
