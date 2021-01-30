from __future__ import annotations

import itertools
import re
import statistics
import typing as t
from collections import defaultdict

import lemminflect
import numpy as np
import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from recap_argument_graph_adaptation.model.config import Config
from scipy.spatial import distance
from spacy.language import Language

# from spacy.tokens import Doc, Span, Token  # type: ignore
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
fuzzymax = config["nlp"]["fuzzymax"]


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

    if embeddings == "integrated" and fuzzymax:
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

        return doc

    def vectors(self, obj):
        return list(itertools.chain(t.vector for t in obj))


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
# ke.textrank, ke.yake, ke.scake, ke.sgrank
# alternative: https://github.com/boudinfl/pke

_vector_cache = {}
Vector = t.Union[t.Tuple[float, ...], t.Tuple[t.Tuple[float, ...], ...]]


def np2tuple(vector: np.ndarray) -> t.Tuple[float, ...]:
    return tuple(vector.tolist())


def _convert_vector(vector: t.Union[np.ndarray, t.List[np.ndarray]]) -> Vector:
    if isinstance(vector, list):
        return tuple(tuple(v.tolist()) for v in vector)
    elif fuzzymax:
        return (tuple(vector.tolist()),)
    else:
        return tuple(vector.tolist())


def _vector(text: str) -> Vector:
    if text not in _vector_cache:
        # with nlp.disable_pipes(vector_disabled_pipes):
        _vector_cache[text] = _convert_vector(nlp(text).vector)

    return _vector_cache[text]


def _vectors(
    texts: t.Iterable[str],
) -> t.Tuple[Vector, ...]:
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

    return tuple(_vector_cache[text] for text in texts)


def _inflect(doc: t.Any, transformation: str) -> str:
    inflections = []
    max_idx = len(doc) - 1

    for i, token in enumerate(doc):
        pos = token.tag_
        inflected = token.text

        if transformation == "singular":
            if i == max_idx:
                if pos == "NNS":
                    inflected = token._.inflect("NN")
                elif pos == "NNPS":
                    inflected = token._.inflect("NNP")

        elif transformation == "plural":
            if i == max_idx:
                if pos == "NN":
                    inflected = token._.inflect("NNS")
                elif pos == "NNP":
                    inflected = token._.inflect("NNPS")

        elif transformation == "lemma":
            inflected = token._.lemma()

        elif transformation == "norm":
            inflected = token.norm_

        else:
            raise ValueError(
                f"The transformation '{transformation}' is unknown. Use either 'singular' or 'plural'."
            )

        inflections.append(inflected)

    return " ".join(inflections)


class InflectionQuery(BaseModel):
    keyword: str
    pos: t.Optional[str]


@dataclass(frozen=True)
class InflectionResponse:
    keyword: str
    vector: Vector
    forms: t.FrozenSet[str]


@app.post("/inflect")
def inflect(query: InflectionQuery) -> InflectionResponse:
    if query.pos == "noun":
        query_kw = f"the {query.keyword} avoided"
        kw = nlp(query_kw)[1:-1]
    else:
        query_kw = query.keyword
        kw = nlp(query_kw)

    lemma = _inflect(kw, "lemma")

    forms = frozenset(
        {
            query.keyword,
            lemma,
            _inflect(kw, "singular"),
            _inflect(kw, "plural"),
            _inflect(kw, "norm"),
        }
    )

    return InflectionResponse(
        keyword=lemma,
        vector=_convert_vector(kw.vector),
        forms=forms,
    )


class VectorQuery(BaseModel):
    text: str


@app.post("/vector")
def vector(query: VectorQuery) -> Vector:
    return _vector(query.text)


class VectorsQuery(BaseModel):
    texts: t.Tuple[str, ...]


@app.post("/vectors")
def vectors(
    query: VectorsQuery,
) -> t.Tuple[Vector, ...]:
    return _vectors(query.texts)


class KeywordQuery(BaseModel):
    texts: t.Tuple[str, ...]
    pos_tags: t.Tuple[str, ...]


@dataclass(frozen=True)
class KeywordResponse:
    keyword: str
    vector: Vector
    forms: t.FrozenSet[str]
    pos_tag: str
    weight: float

    def __eq__(self, other: KeywordResponse) -> bool:
        return (
            self.keyword == other.keyword
            and self.forms == other.forms
            and self.pos_tag == other.pos_tag
            and self.weight == other.weight
        )

    def __hash__(self) -> int:
        return hash((self.keyword, self.forms, self.pos_tag, self.weight))


_keyword_cache = {}


def _dist2sim(distance: t.Optional[float]) -> t.Optional[float]:
    if distance is not None:
        return 1 / (1 + distance)

    return None


@app.post("/keywords", response_model=t.Tuple[t.Tuple[KeywordResponse, ...], ...])
def keywords(query: KeywordQuery) -> t.Tuple[t.Tuple[KeywordResponse, ...], ...]:
    unknown_texts = []

    for text in query.texts:
        if text not in _keyword_cache:
            unknown_texts.append(text)

    if unknown_texts:
        docs: t.Iterable[t.Any] = nlp.pipe(unknown_texts)

        for text, doc in zip(unknown_texts, docs):
            doc_keywords = []

            for pos_tag in query.pos_tags:
                # https://github.com/chartbeat-labs/textacy/blob/cdedd2351bf2a56e8773ec162e08c3188809d486/src/textacy/ke/yake.py#L137
                # Textacy uses token.norm_ if normalize is None.
                # This causes for example 'centres' to be extracted as 'centers'.
                # To avoid this, we use 'lower' instead.
                keywords = extractor(
                    doc, include_pos=pos_tag, normalize="lower", topn=1.0
                )
                processed_keywords = list(nlp.pipe(kw for kw, _ in keywords))

                keyword_vectors = {}
                keyword_scores = {}
                keyword_forms = defaultdict(set)
                lemmas = set()

                for (plain_kw, score), kw in zip(keywords, processed_keywords):
                    lemma = _inflect(kw, "lemma")
                    lemmas.add(lemma)
                    keyword_vectors[lemma] = _convert_vector(kw.vector)
                    keyword_scores[lemma] = score

                    forms = {
                        lemma,
                        plain_kw,
                        _inflect(kw, "singular"),
                        _inflect(kw, "plural"),
                        _inflect(kw, "norm"),
                    }

                    # TODO: Execute this filtering step on the client side
                    for form in forms:
                        pattern = re.compile(f"\\b({form})\\b")

                        if pattern.search(text):
                            keyword_forms[lemma].add(form)

                for lemma in lemmas:
                    forms = keyword_forms.get(lemma)
                    score = keyword_scores[lemma]
                    vector = keyword_vectors[lemma]

                    if forms is not None:
                        doc_keywords.append(
                            KeywordResponse(
                                keyword=lemma,
                                vector=vector,
                                forms=frozenset(forms),
                                pos_tag=pos_tag,
                                weight=_dist2sim(score),
                            )
                        )

            _keyword_cache[text] = tuple(dict.fromkeys(doc_keywords))

    return tuple(_keyword_cache[text] for text in query.texts)


@app.get("/")
def ready() -> str:
    return ""
