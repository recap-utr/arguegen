import typing as t

import numpy as np
import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from spacy.language import Language
from textacy import ke

from .model.config import Config

config = Config.instance()


spacy_cache = {}
proof_reader_cache = {}
spacy_models = {
    "de-integrated": "de_core_news_lg",
    "de-transformers": "de_core_news_sm",
    "en-integrated": "en_core_web_lg",
    "en-transformers": "en_core_web_sm",
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

        if embeddings == "transformers":
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
        if np.any(obj1) and np.any(obj2):
            return 1 - distance.cosine(obj1.vector, obj2.vector)

        return 0.0


app = FastAPI()
nlp = spacy_nlp()
extractor = ke.yake
# ke.textrank, ke.yake, ke.scake, ke.sgrank


class VectorQuery(BaseModel):
    text: str


@app.post("/vector")
def vector(query: VectorQuery) -> t.List[float]:
    return nlp(query.text).vector.tolist()


class SimilarityQuery(BaseModel):
    text1: str
    text2: str


@app.post("/similarity")
def similarity(query: SimilarityQuery) -> float:
    return float(nlp(query.text1).similarity(nlp(query.text2)))


class KeywordQuery(BaseModel):
    text: str
    pos_tags: t.Iterable[str]


@app.post("/keywords")
def keywords(query: KeywordQuery) -> t.List[t.Tuple[str, str, str, float]]:
    doc = nlp(query.text)
    results = []

    for pos_tag in query.pos_tags:
        term_keywords = extractor(doc, include_pos=pos_tag, normalize=None)
        lemma_keywords = extractor(doc, include_pos=pos_tag, normalize="lemma")

        results.extend(
            (
                (term, lemma, pos_tag, weight)
                for (term, weight), (lemma, _) in zip(term_keywords, lemma_keywords)
            )
        )

    return results


@app.get("/")
def ready() -> bool:
    return True
