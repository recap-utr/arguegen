from __future__ import annotations

import itertools
import typing as t
from collections import deque
from dataclasses import dataclass, field
from pprint import pprint

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

from recap_argument_graph_adaptation import wordnet_server
from recap_argument_graph_adaptation.controller import wordnet

s1 = wn.synset("social_group.n.01")
s2 = wn.synset("train.n.01")

# print(_shortest_path_distance(s1.name(), s2.name()))
# print(s1.shortest_path_distance(s2))

print(True)
