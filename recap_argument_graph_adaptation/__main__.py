from pathlib import Path

import nltk
import recap_argument_graph as ag
import typing as t

from .database import Database
from .config import Config

config = Config.instance()


def concepts_in_graph(graph: ag.Graph) -> t.Set[str]:
    concepts = set()

    for node in graph.inodes:
        pos_tags = nltk.pos_tag(nltk.word_tokenize(node.plain_text))

        for word, tag in pos_tags:
            if tag.startswith("N"):
                concepts.add(word)

    return concepts


graph = ag.Graph.open(Path("data/case-base/nodeset6366.json"))
concepts = concepts_in_graph(graph)

db = Database("en")
concept_paths = {}

for concept1 in concepts:
    for concept2 in concepts:
        if concept1 != concept2:
            concept_paths[(concept1, concept2)] = db.get_shortest_path(
                concept1, concept2
            )

print(concept_paths)
