from __future__ import annotations

import itertools
import typing as t

import numpy as np
import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from recap_argument_graph_adaptation.controller.inflect import inflect_concept
from recap_argument_graph_adaptation.model.config import Config
from scipy.spatial import distance
from textacy.extract.keyterms.yake import yake

config = Config.instance()


_models = {
    "en-glove": "en_core_web_lg",
    "en-sbert": "roberta-large-nli-stsb-mean-tokens",
    "en-use": "https://tfhub.dev/google/universal-sentence-encoder-large/5",
}
_backup_models = {
    "en": "en_core_web_sm",
}
fuzzymax = config["nlp"]["fuzzymax"]


def spacy_nlp():
    lang = config["nlp"]["lang"]
    embeddings = config["nlp"]["embeddings"]
    model_name = f"{lang}-{embeddings}"

    spacy_model_name = (
        _models[model_name] if embeddings == "glove" else _backup_models[lang]
    )

    model = spacy.load(
        spacy_model_name, exclude=["ner", "textcat", "parser"]
    )  # parser needed for noun chunks
    model.enable_pipe("senter")

    if embeddings == "glove" and fuzzymax:
        model.add_pipe("fuzzymax", last=True)
    elif embeddings == "sbert":
        model.add_pipe("sbert", last=True, config={"model": model_name})
    elif embeddings == "use":
        model.add_pipe("use", last=True, config={"model": model_name})

    return model


@spacy.Language.component("fuzzymax")
class FuzzyModel:
    def __call__(self, doc):
        doc.user_hooks["vector"] = self.vectors
        doc.user_span_hooks["vector"] = self.vectors

        return doc

    def vectors(self, obj):
        return [t.vector for t in obj]


if config["nlp"]["embeddings"] == "sbert":
    from sentence_transformers import SentenceTransformer

    # https://spacy.io/usage/processing-pipelines#custom-components-user-hooks
    # https://github.com/explosion/spaCy/issues/3823
    @spacy.Language.factory("sbert")
    class TransformerModel:
        def __init__(self, nlp, name, model):
            self._model = SentenceTransformer(_models[model])

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

    @spacy.Language.factory("use")
    class UseModel:
        def __init__(self, nlp, name, model):
            self._model = hub.load(_models[model])

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
# ke.textrank, ke.yake, ke.scake, ke.sgrank
# alternative: https://github.com/boudinfl/pke

_vector_cache = {}
Vector = t.Union[t.Tuple[float, ...], t.Tuple[t.Tuple[float, ...], ...]]


def _convert_vector(vector: t.Union[np.ndarray, t.List[np.ndarray]]) -> Vector:
    if isinstance(vector, (list, tuple)):
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

    unknown_texts = [text for text in texts if text not in _vector_cache]

    if unknown_texts:
        docs = nlp.pipe(unknown_texts)

        for text, doc in zip(unknown_texts, docs):
            _vector_cache[text] = _convert_vector(doc.vector)  # type: ignore

    return tuple(_vector_cache[text] for text in texts)


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


class KeywordResponse(BaseModel):
    keyword: str
    vector: Vector
    form2pos: t.Dict[str, t.Tuple[str, ...]]
    pos2form: t.Dict[str, t.Tuple[str, ...]]
    pos_tag: str
    weight: float


_keyword_cache = {}


def _dist2sim(distance: t.Optional[float]) -> t.Optional[float]:
    if distance is not None:
        return 1 / (1 + distance)

    return None


@app.post("/keywords", response_model=t.Tuple[KeywordResponse, ...])
def keywords(query: KeywordQuery) -> t.Tuple[KeywordResponse, ...]:
    unknown_combinations = [
        (text, pos)
        for text, pos in itertools.product(query.texts, query.pos_tags)
        if (text, pos) not in _keyword_cache
    ]

    if unknown_combinations:
        docs: t.Iterable[t.Any] = nlp.pipe([x[0] for x in unknown_combinations])

        for (text, pos_tag), doc in zip(unknown_combinations, docs):
            doc_keywords = []

            # https://github.com/chartbeat-labs/textacy/blob/cdedd2351bf2a56e8773ec162e08c3188809d486/src/textacy/ke/yake.py#L137
            # Textacy uses token.norm_ if normalize is None.
            # This causes for example 'centres' to be extracted as 'centers'.
            # To avoid this, we use 'lower' instead.
            keywords = yake(doc, include_pos=pos_tag, normalize="lower", topn=1.0)
            # processed_keywords = list(nlp.pipe(kw for kw, _ in keywords))

            for kw_, score in keywords:
                lemma, form2pos, pos2form = inflect_concept(kw_, pos_tag)
                vector = _vector(lemma)

                if form2pos is not None and pos2form is not None:
                    doc_keywords.append(
                        KeywordResponse(
                            keyword=lemma,
                            vector=vector,
                            form2pos=form2pos,
                            pos2form=pos2form,
                            pos_tag=pos_tag,
                            weight=_dist2sim(score),
                        )
                    )

            _keyword_cache[(text, pos_tag)] = tuple(
                doc_keywords
            )  # tuple(dict.fromkeys(doc_keywords))

    return tuple(
        itertools.chain.from_iterable(
            _keyword_cache[(text, pos)]
            for text, pos in itertools.product(query.texts, query.pos_tags)
        )
    )


@app.get("/")
def ready() -> str:
    return ""
