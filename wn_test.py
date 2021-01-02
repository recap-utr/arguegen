from __future__ import annotations

import itertools
import timeit
import typing as t
from collections import deque
from dataclasses import dataclass, field
from pprint import pprint

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

# from recap_argument_graph_adaptation import wordnet_server
# from recap_argument_graph_adaptation.controller import wordnet

print(
    sum(
        timeit.repeat(
            "wn.synsets('dog'); wn.synset('dog.n.01').definition()",
            setup="from nltk.corpus import wordnet as wn; wn.ensure_loaded()",
            number=100,
        )
    )
)

print(
    sum(
        timeit.repeat(
            # "session.post('http://0.0.0.0:8766/synset/definition', json={'code': 'dog.n.01'}).text",
            # setup="""import requests; session = requests.Session()""",
            "wordnet_server.concept_synsets('dog'); wordnet_server.definition['dog.n.01']",
            "from recap_argument_graph_adaptation import wordnet_server",
            number=100,
        )
    )
)

# print(
#     sum(
#         timeit.repeat(
#             "nlp('dog').vector",
#             setup="""import spacy; nlp = spacy.load('en_core_web_lg');""",
#             number=1,
#         )
#     )
# )

# print(
#     sum(
#         timeit.repeat(
#             "spacy.vector('dog')",
#             setup="""from recap_argument_graph_adaptation.controller import spacy""",
#             number=1,
#         )
#     )
# )
