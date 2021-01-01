from __future__ import annotations

import typing as t
from collections import deque
from dataclasses import dataclass, field
from pprint import pprint

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

from recap_argument_graph_adaptation.wordnet_server import hypernyms

s1 = wn.synset("social_group.n.01")
s2 = wn.synset("train.n.01")


def _shortest_path_distance(code1: str, code2: str) -> float:
    """https://www.nltk.org/api/nltk.corpus.reader.html#nltk.corpus.reader.wordnet.Synset.shortest_path_distance"""

    if code1 == code2:
        return 0

    criterion = (code1, code2)

    inf = float("inf")
    path_distance = inf
    queue = deque([hypernyms[code1] + hypernyms[code2]])
    i = 1

    while path_distance == inf and len(queue) > 0:
        candidates = queue.pop()
        new_hypernyms = []

        for candidate in candidates:
            new_hypernyms.extend(hypernyms[candidate])

            if candidate in criterion:
                path_distance = i + 1

        if new_hypernyms:
            queue.append(new_hypernyms)

        i += 1

    return path_distance


print(_shortest_path_distance(s1.name(), s2.name()))
print(s1.shortest_path_distance(s2))

print(True)
