import typing as t

import numpy as np
import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from scipy.spatial import distance
from spacy.language import Language
from textacy import ke

from .model.config import Config

config = Config.instance()


spacy_cache = {}
proof_reader_cache = {}
spacy_models = {
    "de-spacy": "de_core_news_lg",
    "de-transformers": "de_core_news_sm",
    "en-spacy": "en_core_web_lg",
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

if config["nlp"]["embeddings"] == "transformers":
    from sentence_transformers import SentenceTransformer

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
vector_disabled_pipes = ["tagger"]
_vector_cache = {}
# ke.textrank, ke.yake, ke.scake, ke.sgrank


class VectorQuery(BaseModel):
    text: str


@app.post("/vector")
def vector(query: VectorQuery) -> t.List[float]:
    if query.text not in _vector_cache:
        # with nlp.disable_pipes(vector_disabled_pipes):
        _vector_cache[query.text] = nlp(query.text).vector.tolist()

    return _vector_cache[query.text]


class VectorsQuery(BaseModel):
    texts: t.List[str]


@app.post("/vectors")
def vectors(query: VectorsQuery) -> t.List[t.List[float]]:
    docs = nlp.pipe(query.texts)  # disable=vector_disabled_pipes

    return [doc.vector.tolist() for doc in docs]  # type: ignore


class KeywordQuery(BaseModel):
    texts: t.List[str]
    pos_tags: t.List[str]


class KeywordResponse(BaseModel):
    term: str
    lemma: str
    pos_tag: str
    weight: float


@app.post("/keywords", response_model=t.List[t.List[KeywordResponse]])
def keywords(query: KeywordQuery) -> t.List[t.List[KeywordResponse]]:
    docs = nlp.pipe(query.texts)
    response = []

    for doc in docs:
        doc_keywords = []

        for pos_tag in query.pos_tags:
            term_keywords = extractor(doc, include_pos=pos_tag, normalize=None)
            lemma_keywords = extractor(doc, include_pos=pos_tag, normalize="lemma")

            for (term, weight), (lemma, _) in zip(term_keywords, lemma_keywords):
                doc_keywords.append(
                    KeywordResponse(
                        term=term, lemma=lemma, pos_tag=pos_tag, weight=weight
                    )
                )

        response.append(doc_keywords)

    return response


@app.get("/")
def ready() -> str:
    return ""
