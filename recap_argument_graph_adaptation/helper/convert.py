import typing as t
from recap_argument_graph_adaptation.model.adaptation import Concept


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