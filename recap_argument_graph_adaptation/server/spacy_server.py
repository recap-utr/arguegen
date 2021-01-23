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


def _vector(text: str) -> t.List[float]:
    if text not in _vector_cache:
        # with nlp.disable_pipes(vector_disabled_pipes):
        _vector_cache[text] = nlp(text).vector.tolist()

    return _vector_cache[text]


def _vectors(texts: t.Iterable[str]) -> t.List[t.List[float]]:
    # docs = nlp.pipe(query.texts)  # disable=vector_disabled_pipes
    # return [doc.vector.tolist() for doc in docs]  # type: ignore

    unknown_texts = []

    for text in texts:
        if text not in _vector_cache:
            unknown_texts.append(text)

    if unknown_texts:
        docs = nlp.pipe(unknown_texts)

        for text, doc in zip(unknown_texts, docs):
            _vector_cache[text] = doc.vector.tolist()  # type: ignore

    return [_vector_cache[text] for text in texts]


class VectorQuery(BaseModel):
    text: str


@app.post("/vector")
def vector(query: VectorQuery) -> t.List[float]:
    return _vector(query.text)


class VectorsQuery(BaseModel):
    texts: t.List[str]


@app.post("/vectors")
def vectors(query: VectorsQuery) -> t.List[t.List[float]]:
    return _vectors(query.texts)


class KeywordQuery(BaseModel):
    texts: t.List[str]
    pos_tags: t.List[str]


# class KeywordResponse(BaseModel):
#     term: str
#     vector: t.List[float]
#     lemma: str
#     pos_tag: str
#     weight: float


# class KeywordsResponse(BaseModel):
#     keywords: t.List[KeywordResponse]
#     vector: t.List[float]


# @app.post("/keywords", response_model=t.List[KeywordsResponse])
# def keywords(query: KeywordQuery) -> t.List[KeywordsResponse]:
#     docs: t.Iterable[t.Any] = nlp.pipe(query.texts)
#     response = []

#     for doc in docs:
#         doc_keywords = []

#         for pos_tag in query.pos_tags:
#             term_keywords = extractor(doc, include_pos=pos_tag, normalize=None)
#             lemma_keywords = extractor(doc, include_pos=pos_tag, normalize="lemma")

#             for (term, weight), (lemma, _) in zip(term_keywords, lemma_keywords):
#                 doc_keywords.append(
#                     KeywordResponse(
#                         term=term,
#                         vector=_vector(term),
#                         lemma=lemma,
#                         pos_tag=pos_tag,
#                         weight=weight,
#                     )
#                 )

#         response.append(
#             KeywordsResponse(keywords=doc_keywords, vector=doc.vector.tolist())
#         )

#     return response


class KeywordResponse(BaseModel):
    term: str
    vector: t.List[float]
    lemma: str
    pos_tag: str
    weight: float


@dataclass(frozen=True)
class KeywordCandidate:
    term: str
    vector: t.List[float]
    lemma: str
    pos_tag: str

    def __eq__(self, other) -> bool:
        return (
            self.term == other.term
            and self.lemma == other.lemma
            and self.pos_tag == other.pos_tag
        )

    def __hash__(self) -> int:
        return hash((self.term, self.lemma, self.pos_tag))


@app.post("/keywords", response_model=t.List[KeywordResponse])
def keywords(query: KeywordQuery) -> t.List[KeywordResponse]:
    docs: t.Iterable[t.Any] = nlp.pipe(query.texts)
    weights = defaultdict(list)

    for doc in docs:
        for pos_tag in query.pos_tags:
            term_keywords = extractor(doc, include_pos=pos_tag, normalize=None)
            lemma_keywords = extractor(doc, include_pos=pos_tag, normalize="lemma")

            for (term, weight), (lemma, _) in zip(term_keywords, lemma_keywords):
                # TODO: Some terms change their spelling, e.g. centres is extracted as (term: centers, lemma: centre)
                if term in doc.text.lower():
                    candidate = KeywordCandidate(
                        term=term,
                        vector=_vector(term),
                        lemma=lemma,
                        pos_tag=pos_tag,
                    )

                    weights[candidate].append(weight)

    return [
        KeywordResponse(
            term=key.term,
            vector=key.vector,
            lemma=key.lemma,
            pos_tag=key.pos_tag,
            weight=statistics.mean(values),
        )
        for key, values in weights.items()
    ]


@app.get("/")
def ready() -> str:
    return ""
