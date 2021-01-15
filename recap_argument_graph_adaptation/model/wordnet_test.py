from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset
from recap_argument_graph_adaptation.model import wordnet as recap_wn

s: Synset = wn.synset("dog.n.01")
print(s.hypernyms())
print(s.hypernym_distances())
print(s.hypernym_paths())
