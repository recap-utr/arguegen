import typing as t

from recap_argument_graph_adaptation.model import graph
from recap_argument_graph_adaptation.model.adaptation import Concept


def statistic(
    concepts: t.Iterable[Concept],
    reference_paths: t.Mapping[Concept, t.Iterable[graph.Path]],
    adapted_concepts: t.Mapping[Concept, Concept],
    adapted_paths: t.Mapping[Concept, t.Iterable[graph.Path]],
) -> t.Dict[str, t.Any]:
    out = {}

    for concept in concepts:
        key = f"({concept})->({adapted_concepts.get(concept)})"

        out[key] = {
            "concept": str(concept),
            "nodes": list_str(concept.nodes),
            "reference_paths": list_str(reference_paths.get(concept)),
            "adapted_paths": list_str(adapted_paths.get(concept)),
            "adapted_name": str(adapted_concepts.get(concept)),
        }

    return out


def list_str(items: t.Optional[t.Iterable[t.Any]]) -> t.Optional[t.List[str]]:
    """Convert a list of items into a list of strings to serialize as json"""

    return [str(item) for item in items] if items else None
