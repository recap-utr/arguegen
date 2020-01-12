import typing as t
from dataclasses import dataclass
from pathlib import Path

import neo4j
import recap_argument_graph as ag
import spacy
import stackprinter

from . import conceptnet
from .config import Config
from .database import Database

config = Config.instance()
db = Database("en")
nlp = spacy.load("en")

import stackprinter

stackprinter.set_excepthook(style="darkbg")


def concepts_in_graph(graph: ag.Graph) -> t.Set[str]:
    concepts = set()

    for node in graph.inodes:
        for token in node.text:
            if token.tag_.startswith("N") and not token.is_stop:
                concepts.add(token.text)

        # for chunk in node.text.noun_chunks:
        #     concepts.add(chunk.root.text)

    return concepts


@dataclass(frozen=True)
class ConceptAdaptation:
    adapt_from: str
    adapt_to: str
    name: str


def concept_paths(
    concepts: t.Iterable[str], adaptation_rules: t.Iterable[t.Tuple[str, str]]
) -> t.Dict[ConceptAdaptation, t.Optional[neo4j.Path]]:
    paths = {}

    for adapt_from, adapt_to in adaptation_rules:
        for concept in concepts:
            if adapt_from != concept:
                concept_adaptation = ConceptAdaptation(adapt_from, adapt_to, concept)
                paths[concept_adaptation] = db.shortest_path(adapt_from, concept)

    return paths


case_graph = ag.Graph.open(Path("data/case-base/nodeset6366.json"), nlp)
adapted_graph = ag.Graph.open(Path("data/case-base/nodeset6366.json"))
query_graph = ag.Graph("Query")
query_graph.add_node(
    ag.Node(
        query_graph.keygen(),
        nlp(
            "Any punishment is a legal means that as such is not practicable in Germany."
        ),
        ag.NodeCategory.I,
    )
)

case_concepts = concepts_in_graph(case_graph)
query_concepts = concepts_in_graph(query_graph)

# TODO: How to deal with empty replacements: death -> None
# TODO: How to determine multi-token concepts: death_penalty
# TODO: allShortestPaths could be used to filter for paths that contain names of existing concepts in graph
# TODO: Selection of the final adapted relationship is random: next()

adaptation_rules = [("penalty", "punishment"), ("death", "any")]
original_paths = concept_paths(case_concepts, adaptation_rules)
adapted_paths = {}

for concept_adaptation, original_path in original_paths.items():
    adapted_path = db.single_path(concept_adaptation.adapt_to)

    for original_relationship in original_path:
        adapted_relationships = db.expand_node(
            adapted_path.end_node, [original_relationship.type]
        )

        adapted_relationship = next(iter(adapted_relationships), None)

        if adapted_path and adapted_relationship:
            adapted_path = db.extend_path(adapted_path, adapted_relationship)

    adapted_paths[concept_adaptation] = adapted_path


adaptation_rule = adaptation_rules[0]
for concept_adaptation, adapted_path in adapted_paths.items():
    if (
        concept_adaptation.adapt_from == adaptation_rule[0]
        and concept_adaptation.adapt_to == adaptation_rule[1]
    ):
        for node in adapted_graph.inodes:
            node.text = node.text.replace(
                concept_adaptation.name,
                conceptnet.adapt_name(
                    adapted_path.end_node["name"], concept_adaptation.name
                ),
            )

adapted_graph.render(Path("data/out/"))
