import itertools
import statistics
import typing as t
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from recap_argument_graph_adaptation.model.config import Config
from scipy.spatial import distance
from spacy.language import Language
from textacy import ke

config = Config.instance()


_models = {
    "en-integrated": "en_core_web_lg",
    "en-sbert": "roberta-large-nli-stsb-mean-tokens",
    "en-use": "https://tfhub.dev/google/universal-sentence-encoder-large/5",
}
_backup_models = {
    "en": "en_core_web_sm",
}


def spacy_nlp() -> Language:
    lang = config["nlp"]["lang"]
    embeddings = config["nlp"]["embeddings"]
    model_name = f"{lang}-{embeddings}"

    spacy_model_name = (
        _models[model_name] if embeddings == "integrated" else _backup_models[lang]
    )

    model = spacy.load(
        spacy_model_name, disable=["ner", "textcat", "parser"]
    )  # parser needed for noun chunks
    model.add_pipe(model.create_pipe("sentencizer"))

    if embeddings == "integrated" and config["nlp"]["fuzzymax"]:
        model.add_pipe(FuzzyModel(), first=True)
    if embeddings == "sbert":
        model.add_pipe(TransformerModel(model_name), first=True)
    elif embeddings == "use":
        model.add_pipe(UseModel(model_name), first=True)

    return model


class FuzzyModel:
    def __call__(self, doc):
        doc.user_hooks["vector"] = self.vectors
        doc.user_span_hooks["vector"] = self.vectors
        doc.user_token_hooks["vector"] = self.token_vectors

        return doc

    def vectors(self, obj):
        return itertools.chain(t.vector for t in obj)

    def token_vectors(self, obj):
        # https://github.com/explosion/spaCy/blob/5ace559201c714ab89b3092b87d791e16973f31d/spacy/tokens/token.pyx#L387
        if obj.vocab.vectors.size == 0 and obj.doc.tensor.size != 0:
            return [obj.doc.tensor[obj.i]]
        else:
            return [obj.vocab.get_vector(obj.c.lex.orth)]


if config["nlp"]["embeddings"] == "sbert":
    from sentence_transformers import SentenceTransformer

    # https://spacy.io/usage/processing-pipelines#custom-components-user-hooks
    # https://github.com/explosion/spaCy/issues/3823
    class TransformerModel:
        def __init__(self, lang):
            self._model = SentenceTransformer(_models[lang])

        def __call__(self, doc):
            doc.user_hooks["vector"] = self.vector
            doc.user_span_hooks["vector"] = self.vector
            doc.user_token_hooks["vector"] = self.vector

            return doc

        def vector(self, obj):
            sentences = [obj.text]
            embeddings = self._model.encode(sentences)

            return embeddings[0]


if config["nlp"]["embeddings"] == "use":
    import tensorflow_hub as hub

    class UseModel:
        def __init__(self, lang):
            self._model = hub.load(_models[lang])

        def __call__(self, doc):
            doc.user_hooks["vector"] = self.vector
            doc.user_span_hooks["vector"] = self.vector
            doc.user_token_hooks["vector"] = self.vector

            return doc

        def vector(self, obj):
            sentences = [obj.text]
            embeddings = self._model(sentences)  # type: ignore

            return embeddings[0].numpy()


app = FastAPI()
nlp = spacy_nlp()
extractor = ke.yake
_vector_cache = {}
Vector = t.Union[t.List[float], t.List[t.List[float]]]
# ke.textrank, ke.yake, ke.scake, ke.sgrank


def _convert_vector(vector: t.Union[np.ndarray, t.List[np.ndarray]]) -> Vector:
    if isinstance(vector, list):
        return [v.tolist() for v in vector]
    else:
        return vector.tolist()


def _vector(text: str) -> Vector:
    if text not in _vector_cache:
        # with nlp.disable_pipes(vector_disabled_pipes):
        _vector_cache[text] = nlp(text).vector

    return _vector_cache[text]


def _vectors(
    texts: t.Iterable[str],
) -> t.List[Vector]:
    # docs = nlp.pipe(query.texts)  # disable=vector_disabled_pipes
    # return [doc.vector.tolist() for doc in docs]  # type: ignore

    unknown_texts = []

    for text in texts:
        if text not in _vector_cache:
            unknown_texts.append(text)

    if unknown_texts:
        docs = nlp.pipe(unknown_texts)

        for text, doc in zip(unknown_texts, docs):
            _vector_cache[text] = _convert_vector(doc.vector)  # type: ignore

    return [_vector_cache[text] for text in texts]


class VectorQuery(BaseModel):
    text: str


@app.post("/vector")
def vector(query: VectorQuery) -> Vector:
    return _vector(query.text)


class VectorsQuery(BaseModel):
    texts: t.List[str]


@app.post("/vectors")
def vectors(
    query: VectorsQuery,
) -> t.List[Vector]:
    return _vectors(query.texts)


class KeywordQuery(BaseModel):
    texts: t.List[str]
    pos_tags: t.List[str]


class KeywordResponse(BaseModel):
    term: str
    vector: Vector
    lemma: str
    pos_tag: str
    weight: float


_keyword_cache = {}


@app.post("/keywords", response_model=t.List[t.List[KeywordResponse]])
def keywords(query: KeywordQuery) -> t.List[t.List[KeywordResponse]]:
    unknown_texts = []

    for text in query.texts:
        if text not in _keyword_cache:
            unknown_texts.append(text)

    if unknown_texts:
        docs: t.Iterable[t.Any] = nlp.pipe(unknown_texts)

        for text, doc in zip(unknown_texts, docs):
            doc_keywords = []

            for pos_tag in query.pos_tags:
                term_keywords = extractor(doc, include_pos=pos_tag, normalize=None)
                lemma_keywords = extractor(doc, include_pos=pos_tag, normalize="lemma")

                for (term, weight), (lemma, _) in zip(term_keywords, lemma_keywords):
                    # TODO: Some terms change their spelling, e.g. centres is extracted as (term: centers, lemma: centre)
                    if term in doc.text.lower():
                        doc_keywords.append(
                            KeywordResponse(
                                term=term,
                                vector=_vector(term),
                                lemma=lemma,
                                pos_tag=pos_tag,
                                weight=weight,
                            )
                        )

            _keyword_cache[text] = doc_keywords

    return [_keyword_cache[text] for text in query.texts]


@app.get("/")
def ready() -> str:
    return ""
