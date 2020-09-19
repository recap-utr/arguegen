import csv
import logging
import warnings

import spacy
import typing as t
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import lmproof

import recap_argument_graph as ag
from recap_argument_graph_adaptation.model import adaptation
from recap_argument_graph_adaptation.model.config import config


def cases() -> t.List[adaptation.Case]:
    input_path = Path(config["path"]["input"])
    result = []

    for folder in sorted(input_path.iterdir()):
        if folder.is_dir():
            result.append(_case(folder))

    if not result:  # no nested folders were found
        result.append(_case(input_path))

    return result


def _case(path: Path) -> adaptation.Case:
    input_graph = ag.Graph.open(path / "case.json")
    input_rules = _parse_rules(path / "case.csv")

    benchmark_graph = ag.Graph.open(path / "benchmark.json")
    benchmark_rules = _parse_rules(path / "benchmark.csv")

    with (path / "query.txt").open() as file:
        query = file.read()

    return adaptation.Case(
        path.name,
        query,
        input_graph,
        input_rules,
        benchmark_graph,
        benchmark_rules,
    )


def _parse_rules(path: Path) -> t.List[adaptation.Rule]:
    rules = []

    with path.open() as file:
        reader = csv.reader(file, delimiter=",")

        for row in reader:
            rules.append(adaptation.Rule(row[0], row[1]))

    return rules


spacy_cache = {"en": None, "de": None}
proof_reader_cache = {"en": None, "de": None}
spacy_models = {"en": "en_core_web_sm", "de": "de_core_news_sm"}
transformer_models = {
    "en": "distiluse-base-multilingual-cased",
    "de": "distiluse-base-multilingual-cased",
}


def spacy_nlp():
    lang = config["nlp"]["lang"]

    if not spacy_cache[lang]:
        model = spacy.load(spacy_models[lang])
        model.add_pipe(TransformerModel(lang), first=True)

        spacy_cache[lang] = model

    return spacy_cache[lang]


def proof_reader() -> lmproof.Proofreader:
    lang = config["nlp"]["lang"]

    if not proof_reader_cache[lang]:
        proof_reader_cache[lang] = lmproof.load(lang)

    return proof_reader_cache[lang]


# https://spacy.io/usage/processing-pipelines#custom-components-user-hooks
# https://github.com/explosion/spaCy/issues/3823
class TransformerModel(object):
    def __init__(self, lang):
        self._model = SentenceTransformer(transformer_models[lang])

    def __call__(self, doc):
        doc.user_hooks["vector"] = self.vector
        doc.user_span_hooks["vector"] = self.vector
        doc.user_token_hooks["vector"] = self.vector

        return doc

    def vector(self, obj):
        # The function `encode` expects a list of strings.
        sentences = [obj.text]
        embeddings = self._model.encode(sentences)

        return embeddings[0]
