import csv
import logging
from socketserver import ThreadingMixIn
import typing as t
import warnings
from pathlib import Path
from xmlrpc.server import SimpleXMLRPCServer
import threading

import lmproof
import recap_argument_graph as ag
import spacy
from textacy import ke
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
    lang = "en"  # TODO: config["nlp"]["lang"]
    embeddings = "integrated"  # TODO: config["nlp"]["embeddings"]
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


nlp = spacy_nlp()
extractor = ke.yake
# ke.textrank, ke.yake, ke.scake, ke.sgrank

# https://stackoverflow.com/questions/53621682/multi-threaded-xml-rpc-python3-7-1
class SimpleThreadedXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass


# https://github.com/dcavar/spacyxmlrpc/blob/master/spacyRPCServer.py
# server = SimpleXMLRPCServer(("localhost", 8000), allow_none=True, logRequests=False)
server = SimpleThreadedXMLRPCServer(
    ("localhost", 8000), allow_none=True, logRequests=False
)
server.register_introspection_functions()

# Register a function under function.__name__.
@server.register_function()
def vector(text: str) -> t.List[float]:
    return nlp(text).vector.tolist()


@server.register_function()
def similarity(text1: str, text2: str) -> float:
    return float(nlp(text1).similarity(nlp(text2)))


@server.register_function()
def keywords(
    text: str, pos_tags: t.Iterable[str], normalize: bool
) -> t.List[t.Tuple[str, str, float]]:
    doc = nlp(text)
    terms = []
    normalize_func = "lemma" if normalize else None

    for pos_tag in pos_tags:
        keywords = extractor(doc, include_pos=pos_tag, normalize=normalize_func)
        terms.extend(((keyword, pos_tag, weight) for keyword, weight in keywords))

    return terms


@server.register_function()
def ready() -> bool:
    return True


print("Ready.")
server.serve_forever()

# https://stackoverflow.com/questions/15074393/python-multithreading-xmlrpc-server
# server_thread = threading.Thread(target=server.serve_forever)
# server_thread.start()
