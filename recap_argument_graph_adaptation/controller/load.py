import csv

import spacy
import typing as t
from pathlib import Path
from sentence_transformers import SentenceTransformer

import recap_argument_graph as ag
from recap_argument_graph_adaptation.model import adaptation
from recap_argument_graph_adaptation.model.config import config


def cases() -> t.List[adaptation.Case]:
    case_folder = Path(config["path"]["input"])
    result = []

    for graph_file in sorted(case_folder.rglob("*.json")):
        graph = ag.Graph.open(graph_file)
        rule_file = graph_file.with_suffix(".csv")

        if rule_file.is_file():
            rules = _parse_rules(rule_file)

            result.append(adaptation.Case(graph=graph, rules=rules))

    return result


def _parse_rules(path: Path) -> t.List[adaptation.Rule]:
    rules = []

    with path.open() as file:
        reader = csv.reader(file, delimiter=",")

        for row in reader:
            rules.append((row[0], row[1]))

    return rules


spacy_cache = {"en": None, "de": None}
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
        return self._model.encode([str(obj)])[0]
