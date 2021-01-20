import typing as t


def list_str(items: t.Optional[t.Iterable[t.Any]]) -> t.Optional[t.List[str]]:
    """Convert a list of items into a list of strings to serialize as json"""

    return [str(item) for item in items] if items else None


def list_dict(
    items: t.Optional[t.Iterable[t.Any]], **kwargs
) -> t.Optional[t.List[str]]:
    """Convert a list of items into a list of strings to serialize as json"""

    return [item.to_dict(kwargs) for item in items] if items else None


def xstr(obj: t.Optional[t.Any]) -> t.Optional[str]:
    if obj is not None:
        return str(obj)

    return None
