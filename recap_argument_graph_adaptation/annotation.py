import itertools
import typing as t

from nltk.corpus import wordnet as wn


def _pos2wn(pos: str) -> t.Optional[t.List[str]]:
    if pos == "noun":
        return ["n"]
    elif pos == "verb":
        return ["v"]
    elif pos == "adjective":
        return ["a", "s"]
    elif pos == "adverb":
        return ["r"]

    return None


def hypernyms():
    while True:
        try:
            concept_uri = input("Enter concept in the form 'name.pos': ")
            concept, user_pos = concept_uri.split(".")
            concept = concept.replace(" ", "_").strip()
            wn_pos_tags = _pos2wn(user_pos)

            if not wn_pos_tags:
                raise ValueError(
                    "You must enter a valid pos tag: 'noun', 'verb', 'adjective', 'adverb'."
                )

            synsets = []

            for wn_pos in wn_pos_tags:
                synsets.extend(wn.synsets(concept, wn_pos))

            hypernyms = itertools.chain(
                *[
                    hyp.lemmas()
                    for synset in synsets
                    for hyp, _ in synset.hypernym_distances()
                    if not hyp.name().startswith(concept)
                ]
            )
            lemmas = {lemma.name().replace("_", " ") for lemma in hypernyms}

            print(sorted(lemmas))
        except Exception as e:
            print(e)

        print()


if __name__ == "__main__":
    hypernyms()
