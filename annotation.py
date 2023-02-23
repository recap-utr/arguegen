import itertools
import readline
import typing as t

import wn
from rich.console import Console
from rich.table import Table
from rich.text import Text
from wn.morphy import Morphy

db = wn.Wordnet("oewn:2021", lemmatizer=Morphy())
console = Console()
POS_TAGS = "'noun', 'verb', 'adjective', 'adverb'"
readline.set_history_length(1000)


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
    console.print(
        "This script lists all possible hypernyms/hyponyms for a given concept.",
        "Enter your query in the form 'name/pos'.",
        f"The following pos tags are supported: {POS_TAGS}",
        "Example: 'dog/noun'",
        (
            "Please note: They are sorted alphabetically and not according to their"
            " distance in WordNet."
        ),
        "",
        sep="\n",
    )

    while True:
        try:
            concept_uri = console.input(Text("Input: ", style="bold"))
            concept, user_pos = concept_uri.split("/")
            concept = concept.strip()
            wn_pos_tags = _pos2wn(user_pos.strip())

            if not wn_pos_tags:
                raise ValueError(f"You must enter a valid pos tag: {POS_TAGS}")

            synsets: list[wn.Synset] = []

            for wn_pos in wn_pos_tags:
                synsets.extend(db.synsets(concept, wn_pos))

            if not synsets:
                raise ValueError("The concept you entered does not exist in WordNet.")

            hypernyms = itertools.chain.from_iterable(
                hyp.lemmas() for synset in synsets for hyp in synset.hypernyms()
            )
            hyponyms = itertools.chain.from_iterable(
                hyp.lemmas() for synset in synsets for hyp in synset.hyponyms()
            )

            hypernym_lemmas = sorted(
                {
                    f"{lemma}/{user_pos}"
                    for lemma in hypernyms
                    if not lemma.startswith(concept)
                }
            )
            hyponym_lemmas = sorted(
                {
                    f"{lemma}/{user_pos}"
                    for lemma in hyponyms
                    if not lemma.startswith(concept)
                }
            )

            table = Table(show_header=False, show_lines=True)
            table.add_column("Type")
            table.add_column("Values")

            table.add_row("Hypernyms", ", ".join(hypernym_lemmas))
            table.add_row("Hyponyms", ", ".join(hyponym_lemmas))

            console.print(table)
        except Exception as e:
            console.print(e)

        console.print()


if __name__ == "__main__":
    hypernyms()
