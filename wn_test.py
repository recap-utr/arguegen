from recap_argument_graph_adaptation.model import graph
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
import nltk
import typing as t
from pprint import pprint

from recap_argument_graph_adaptation.controller import wordnet

# How to find the correct synset: Word sense disambiguation
# https://github.com/nltk/nltk/blob/develop/nltk/wsd.py

# wordnet.log_synsets(wordnet.synsets("social_group", graph.POS.NOUN))

s1 = wordnet.synset("social_group.n.01")
print(wordnet.hypernym_trees(s1))
