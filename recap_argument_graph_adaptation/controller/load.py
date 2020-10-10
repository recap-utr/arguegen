import csv
import logging
from recap_argument_graph_adaptation.model.adaptation import Concept
from recap_argument_graph_adaptation.model.database import Database
import warnings

import spacy
from spacy.language import Language
import typing as t
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import lmproof

import recap_argument_graph as ag
from recap_argument_graph_adaptation.model import adaptation, graph
from recap_argument_graph_adaptation.model.config import config


spacy_cache = {"en": None, "de": None}
proof_reader_cache = {"en": None, "de": None}
spacy_models = {"en": "en_core_web_sm", "de": "de_core_news_sm"}
transformer_models = {
    "en": "roberta-large-nli-stsb-mean-tokens",
    "de": "distiluse-base-multilingual-cased",
}


def spacy_nlp() -> Language:
    lang = config["nlp"]["lang"]

    if not spacy_cache[lang]:
        model = spacy.load(spacy_models[lang])
        model.add_pipe(TransformerModel(lang), first=True)

        spacy_cache[lang] = model

    return spacy_cache[lang]  # type: ignore


def proof_reader() -> lmproof.Proofreader:
    lang = config["nlp"]["lang"]

    if not proof_reader_cache[lang]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proof_reader_cache[lang] = lmproof.load(lang)  # type: ignore

    return proof_reader_cache[lang]  # type: ignore


# https://spacy.io/usage/processing-pipelines#custom-components-user-hooks
# https://github.com/explosion/spaCy/issues/3823
class TransformerModel(object):
    def __init__(self, lang):
        self._model = SentenceTransformer(transformer_models[lang])

    def __call__(self, doc):
        doc.user_hooks["vector"] = self.vector
        doc.user_span_hooks["vector"] = self.vector
        doc.user_token_hooks["vector"] = self.vector

        doc.user_hooks["similarity"] = self.similarity
        doc.user_span_hooks["similarity"] = self.similarity
        doc.user_token_hooks["similarity"] = self.similarity

        return doc

    def vector(self, obj):
        # The function `encode` expects a list of strings.
        sentences = [obj.text]
        embeddings = self._model.encode(sentences)

        return embeddings[0]

    def similarity(self, obj1, obj2):
        return 1 - distance.cosine(obj1.vector, obj2.vector)


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
    input_graph = ag.Graph.open(path / "case-graph.json")
    input_rules = _parse_rules(path / "case-rules.csv")

    benchmark_graph = ag.Graph.open(path / "benchmark-graph.json")
    benchmark_rules = _parse_rules(path / "benchmark-rules.csv")

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
            source = _parse_rule_concept(row[0])
            target = _parse_rule_concept(row[1])

            rules.append(adaptation.Rule(source, target))

    return rules


def _parse_rule_concept(rule: str) -> Concept:
    nlp = spacy_nlp()

    rule_parts = rule.split("/")
    name = nlp(rule_parts[0])
    pos = graph.POS.OTHER

    if len(rule_parts) > 1:
        pos = graph.POS(rule_parts[1])
    else:
        spacy_pos = name[0].pos_  # POS tags are only available on token level.
        pos = graph.spacy_pos_mapping[spacy_pos]

    db = Database()
    nodes = db.nodes(name.text, pos)

    if not nodes:
        raise ValueError(f"The rule concept '{name}' cannot be found in ConceptNet.")

    return Concept(name, pos, nodes, 1.0, 0)
