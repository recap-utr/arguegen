import csv
import logging
from recap_argument_graph_adaptation.controller import metrics, wordnet
import typing as t
import warnings
from pathlib import Path

import lmproof
import recap_argument_graph as ag
import spacy
from recap_argument_graph_adaptation.model import adaptation, graph
from recap_argument_graph_adaptation.model.adaptation import Concept
from recap_argument_graph_adaptation.model.config import config
from recap_argument_graph_adaptation.model.database import Database
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from spacy.language import Language

spacy_cache = {}
proof_reader_cache = {}
spacy_models = {
    "de-integrated": "de_core_news_lg",
    "de-transformer": "de_core_news_sm",
    "en-integrated": "en_core_web_lg",
    "en-transformer": "en_core_web_sm",
}
transformer_models = {
    "de": "distiluse-base-multilingual-cased",
    "en": "roberta-large-nli-stsb-mean-tokens",
}


def spacy_nlp() -> Language:
    lang = config["nlp"]["lang"]
    embeddings = config["nlp"]["embeddings"]
    model_name = f"{lang}-{embeddings}"

    if not spacy_cache.get(model_name):
        model = spacy.load(
            spacy_models[model_name], disable=["ner", "textcat", "parser"]
        )  # parser needed for noun chunks
        model.add_pipe(model.create_pipe("sentencizer"))

        if embeddings == "transformer":
            model.add_pipe(TransformerModel(lang), first=True)

        spacy_cache[model_name] = model

    return spacy_cache[model_name]  # type: ignore


def proof_reader() -> lmproof.Proofreader:
    lang = config["nlp"]["lang"]

    if not proof_reader_cache.get(lang):
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


def cases() -> t.List[adaptation.PlainCase]:
    input_path = Path(config["path"]["input"])
    result = []

    for folder in sorted(input_path.iterdir()):
        if folder.is_dir():
            result.append(_case(folder))

    if not result:  # no nested folders were found
        result.append(_case(input_path))

    return result


def _case(path: Path) -> adaptation.PlainCase:
    input_graph = ag.Graph.open(path / "case-graph.json")
    input_rules = _parse_rules(path / "case-rules.csv")

    benchmark_graph = ag.Graph.open(path / "benchmark-graph.json")
    benchmark_rules = _parse_rules(path / "benchmark-rules.csv")

    if not (input_graph and input_rules and benchmark_graph and benchmark_rules):
        raise RuntimeError(
            "Not all required assets ('case-graph.json', 'case-rules.csv', 'benchmark-graph.json', 'benchmark-rules.csv') were found"
        )

    with (path / "query.txt").open() as file:
        query = file.read()

    return adaptation.PlainCase(
        path.name,
        query,
        input_graph,
        input_rules,
        benchmark_graph,
        benchmark_rules,
    )


def _parse_rules(path: Path) -> t.Tuple[adaptation.PlainRule]:
    rules = []

    with path.open() as file:
        reader = csv.reader(file, delimiter=",")

        for row in reader:
            source = _parse_rule_concept(row[0])
            target = _parse_rule_concept(row[1])

            rules.append(adaptation.PlainRule(source, target))

    return tuple(rules)


def _parse_rule_concept(rule: str) -> adaptation.PlainConcept:
    rule_parts = rule.split("/")
    name = rule_parts[0]
    pos = graph.POS.OTHER

    if len(rule_parts) > 1:
        pos = graph.POS(rule_parts[1])

    db = Database()
    nodes = db.nodes(name, pos)
    synsets = wordnet.synsets(name, pos)

    if config["nlp"]["knowledge_graph"] == "conceptnet" and not nodes:
        raise ValueError(f"The rule concept '{name}' cannot be found in ConceptNet.")
    elif config["nlp"]["knowledge_graph"] == "wordnet" and not synsets:
        raise ValueError(
            f"The rule concept '{name}/{pos.value}' cannot be found in WordNet."
        )

    return adaptation.PlainConcept(
        name, pos, nodes, synsets, None, metrics.best_concept_metrics
    )
