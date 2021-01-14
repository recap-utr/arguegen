import typing as t

from recap_argument_graph_adaptation.controller import convert
from recap_argument_graph_adaptation.model import casebase, conceptnet


def statistic(
    concepts: t.Iterable[casebase.Concept],
    reference_paths: t.Mapping[casebase.Concept, t.Iterable[conceptnet.Path]],
    adapted_paths: t.Mapping[casebase.Concept, t.Iterable[conceptnet.Path]],
    adapted_synsets: t.Mapping[casebase.Concept, t.Iterable[casebase.Concept]],
    adapted_concepts: t.Mapping[casebase.Concept, casebase.Concept],
) -> t.Dict[str, t.Any]:
    out = {}

    for concept in concepts:
        key = f"({concept})->({adapted_concepts.get(concept)})"
        adapted_synsets_str = (
            {
                str(item): item.score
                for item in sorted(adapted_synsets.get(concept), key=lambda x: x.score)
            }
            if adapted_synsets
            else None
        )

        out[key] = {
            **concept.to_dict(),
            "reference_paths": convert.list_str(reference_paths.get(concept)),
            "adapted_paths": convert.list_str(adapted_paths.get(concept)),
            "adapted_synsets": adapted_synsets_str,
            "adapted_name": convert.xstr(adapted_concepts.get(concept)),
        }

    return out
