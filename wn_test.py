from __future__ import annotations

import json
import timeit
from pathlib import Path

from nltk.corpus import wordnet as wn
from nltk.corpus.reader import WordNetCorpusReader

wn.ensure_loaded()

# with Path("data", "wn_exceptions.json").open("w") as f:
#     json.dump(wn._exception_map, f)

# with Path("data", "wn_morphy_substitutions.json").open("w") as f:
#     json.dump(WordNetCorpusReader.MORPHOLOGICAL_SUBSTITUTIONS, f)

print(
    sum(
        timeit.repeat(
            "s = wn.synsets('dog')[0].name(); wn.synset(s).definition()",
            setup="from nltk.corpus import wordnet as wn; wn.ensure_loaded()",
            number=1000,
        )
    )
)

print(
    sum(
        timeit.repeat(
            "s = wn.synsets('dog')[0]; s.definition()",
            setup="from nltk.corpus import wordnet as wn; wn.ensure_loaded()",
            number=1000,
        )
    )
)

print(
    sum(
        timeit.repeat(
            "s = db.synsets('dog')[0].id; db.synset(s).definition()",
            setup="import wn; db = wn.Wordnet(lang='en', lexicon='pwn:3.1')",
            number=1000,
        )
    )
)

print(
    sum(
        timeit.repeat(
            "s = db.synsets('dog')[0]; s.definition()",
            setup="import wn; db = wn.Wordnet(lang='en', lexicon='pwn:3.1')",
            number=1000,
        )
    )
)

print(
    sum(
        timeit.repeat(
            "code = wordnet.concept_synsets('dog', None)[0]; wordnet.synset_definition(code)",
            "from recap_argument_graph_adaptation.controller import wordnet",
            number=1000,
        )
    )
)

print(
    sum(
        timeit.repeat(
            "s = wordnet._synsets('dog', None)[0]; s.definition()",
            "from recap_argument_graph_adaptation.controller import wordnet",
            number=1000,
        )
    )
)
