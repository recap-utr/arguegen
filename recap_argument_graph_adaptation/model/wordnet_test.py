from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset

s: Synset = wn.synset("dog.n.01")
print(s.hypernyms())
