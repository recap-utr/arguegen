import typing as t

from recap_argument_graph_adaptation.model import graph
from recap_argument_graph_adaptation.model.adaptation import Concept
from recap_argument_graph_adaptation.helper import convert


def statistic(
    concepts: t.Iterable[Concept],
    reference_paths: t.Mapping[Concept, t.Iterable[graph.Path]],
    adapted_paths: t.Mapping[Concept, t.Iterable[graph.Path]],
    adapted_synsets: t.Mapping[Concept, t.Iterable[Concept]],
    adapted_concepts: t.Mapping[Concept, Concept],
) -> t.Dict[str, t.Any]:
    out = {}

    for concept in concepts:
        key = f"({concept})->({adapted_concepts.get(concept)})"

        out[key] = {
            **concept.to_dict(),
            "reference_paths": convert.list_str(reference_paths.get(concept)),
            "adapted_paths": convert.list_str(adapted_paths.get(concept)),
            "adapted_synsets": convert.concepts_str(adapted_synsets.get(concept)),
            "adapted_name": convert.xstr(adapted_concepts.get(concept)),
        }

    return out
