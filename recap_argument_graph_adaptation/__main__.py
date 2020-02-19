import json
import logging
from pathlib import Path
import typing as t

import recap_argument_graph as ag
import spacy

from .controller import adapt, concept, load
from .model.adaptation import Concept
from .model import graph
from .model.config import config

logging.basicConfig(level=logging.INFO)
logging.getLogger(__package__).setLevel(logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

nlp = spacy.load(config["spacy"]["model"])

# TODO: Concept replacement does not work due to lemmatization of keyword extractor
# TODO: Many adapted paths have shorter lengths and thus are not considered!


def _export_adaptation(
    concepts: t.Iterable[Concept],
    reference_paths: t.Dict[Concept, t.Iterable[graph.Path]],
    adapted_concepts: t.Dict[Concept, str],
    adapted_paths: t.Dict[Concept, t.Iterable[graph.Path]],
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


cases = load.cases()
out_path = Path(
    config["path"]["output"],
    config["adaptation"]["method"],
    config["adaptation"]["selector"],
)

for case in cases:
    log.info(f"Processing '{case.graph.name}'.")

    concepts = concept.from_graph(case.graph)
    log.info(
        f"Found the following concepts: {', '.join((str(concept) for concept in concepts))}"
    )

    adaptation_results = {}

    for rule in case.rules:
        reference_paths = concept.paths(concepts, rule)
        adapted_concepts, adapted_paths = adapt.paths(reference_paths, rule)

        adapt.argument_graph(case.graph, rule, adapted_concepts)
        adaptation_results[f"{rule[0]}->{rule[1]}"] = _export_adaptation(
            concepts, reference_paths, adapted_concepts, adapted_paths
        )

    case.graph.save(out_path)
    case.graph.render(out_path)
    stats_path = out_path / f"{case.graph.name}-stats.json"

    with stats_path.open("w") as file:
        json.dump(
            adaptation_results,
            file,
            ensure_ascii=False,
            indent=4,
            default=lambda x: str(x),
            # default=lambda x: x.__dict__,
        )
