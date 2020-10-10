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
            "node": str(concept.node),
            "reference_paths": list_str(reference_paths.get(concept)),
            "adapted_paths": list_str(adapted_paths.get(concept)),
            "adapted_name": adapted_concepts.get(concept),
        }

    return out


def list_str(paths: t.Optional[t.Iterable[graph.Path]]) -> t.Optional[t.List[str]]:
    """Convert a list of paths into a list of strings to serialize as json"""

    return [str(path) for path in paths] if paths else None
