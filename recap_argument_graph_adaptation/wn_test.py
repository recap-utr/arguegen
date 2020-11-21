from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
import nltk
import typing as t

# How to find the correct synset: Word sense disambiguation
# https://github.com/nltk/nltk/blob/develop/nltk/wsd.py


def log_synsets(synsets: t.Iterable[Synset]) -> None:
    for synset in synsets:
        print(f"Name:       {synset.name()}")
        print(f"Definition: {synset.definition()}")
        print(f"Examples:   {synset.examples()}")
        print()


def synset(text: str) -> Synset:
    return wn.synset(text)


def synsets(text: str) -> t.Tuple[Synset, ...]:
    return tuple(wn.synsets(text))


def hypernym_distances(synset: Synset) -> t.List[Synset]:
    hypernym_tuples: t.Set[t.Tuple[Synset, int]] = synset.hypernym_distances()
    hypernym_tuples_sorted = sorted(hypernym_tuples, key=lambda s: s[1])

    return [hypernym for hypernym, pos in hypernym_tuples_sorted]


s1 = synset("school.n.01")
s2 = synset("uniform.n.01")
s1, s2 = s2, s1

print("hypernyms:", s1.hypernyms())
print("hypernym_distances:", hypernym_distances(s1))
print("common_hypernyms:", s1.common_hypernyms(s2))
print("lowest_common_hypernyms:", s1.lowest_common_hypernyms(s2))
print("lch_similarity:", s1.lch_similarity(s2))
print("wup_similarity:", s1.wup_similarity(s2))
print("shortest_path_distance:", s1.shortest_path_distance(s2))
