import typing as t

from nltk.corpus.reader.wordnet import Synset

from recap_argument_graph_adaptation.model import graph
from recap_argument_graph_adaptation.model.adaptation import Concept


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
            "reference_paths": list_str(reference_paths.get(concept)),
            "adapted_paths": list_str(adapted_paths.get(concept)),
            "adapted_synsets": concepts_str(adapted_synsets.get(concept)),
            "adapted_name": xstr(adapted_concepts.get(concept)),
        }

    return out


def concepts_str(
    items: t.Optional[t.Iterable[Concept]],
) -> t.Optional[t.Mapping[str, float]]:
    if items is not None:
        return {str(item): item.score for item in sorted(items, key=lambda i: i.score)}

    return None


def list_str(items: t.Optional[t.Iterable[t.Any]]) -> t.Optional[t.List[str]]:
    """Convert a list of items into a list of strings to serialize as json"""

    return [str(item) for item in items] if items else None


def xstr(obj: t.Optional[t.Any]) -> t.Optional[str]:
    if obj is not None:
        return str(obj)

    return None
