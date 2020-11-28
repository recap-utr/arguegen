from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
import nltk
import typing as t
from pprint import pprint

from recap_argument_graph_adaptation.controller import wordnet

# How to find the correct synset: Word sense disambiguation
# https://github.com/nltk/nltk/blob/develop/nltk/wsd.py

# log_synsets(synsets("prescription"))

s1 = wordnet.synset("prescription_drug.n.01")
print(wordnet.hypernyms(s1))
