from __future__ import annotations

import statistics
import typing as t
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import requests
from recap_argument_graph_adaptation.model.config import Config
from scipy.spatial import distance

config = Config.instance()

# from sentence_transformers import SentenceTransformer

_base_url = f"http://{config['resources']['spacy']['host']}:{config['resources']['spacy']['port']}"
session = requests.Session()


def _url(*parts: str) -> str:
    return "/".join([_base_url, *parts])


_vector_cache = {}
Vector = t.Union[np.ndarray, t.List[np.ndarray]]


def _convert_vector(vector: t.Union[t.List[t.List[float]], t.List[float]]) -> Vector:
    if any(isinstance(i, float) for i in vector):
        return np.array(vector)
    else:
        return [np.array(v) for v in vector]


def vector(text: str) -> Vector:
    if text not in _vector_cache:
        _vector_cache[text] = _convert_vector(
            session.post(_url("vector"), json={"text": text}).json()
        )

    return _vector_cache[text]


# This function does not use the cache!
def vectors(texts: t.Iterable[str]) -> t.List[Vector]:
    unknown_texts = []

    for text in texts:
        if text not in _vector_cache:
            unknown_texts.append(text)

    if unknown_texts:
        vectors = session.post(_url("vectors"), json={"texts": unknown_texts}).json()

        for text, vector in zip(unknown_texts, vectors):
            _vector_cache[text] = _convert_vector(vector)

    return [_vector_cache[text] for text in texts]


# https://github.com/babylonhealth/fuzzymax/blob/master/similarity/fuzzy.py
def fuzzify(s, u):
    """
    Sentence fuzzifier.
    Computes membership vector for the sentence S with respect to the
    universe U
    :param s: list of word embeddings for the sentence
    :param u: the universe matrix U with shape (K, d)
    :return: membership vectors for the sentence
    """
    f_s = np.dot(s, u.T)
    m_s = np.max(f_s, axis=0)
    m_s = np.maximum(m_s, 0, m_s)
    return m_s


# https://github.com/babylonhealth/fuzzymax/blob/master/similarity/fuzzy.py
def dynamax_jaccard(x, y):
    """
    DynaMax-Jaccard similarity measure between two sentences
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity score between the two sentences
    """
    u = np.vstack((x, y))
    m_x = fuzzify(x, u)
    m_y = fuzzify(y, u)

    m_inter = np.sum(np.minimum(m_x, m_y))
    m_union = np.sum(np.maximum(m_x, m_y))
    return m_inter / m_union


def similarity(
    obj1: t.Union[str, np.ndarray, t.List[np.ndarray]],
    obj2: t.Union[str, np.ndarray, t.List[np.ndarray]],
) -> float:
    if isinstance(obj1, str):
        obj1 = vector(obj1)
    if isinstance(obj2, str):
        obj2 = vector(obj2)

    if isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
        if np.any(obj1) and np.any(obj2):
            return float(1 - distance.cosine(obj1, obj2))
        return 0.0

    elif isinstance(obj1, list) and isinstance(obj2, list):
        return dynamax_jaccard(obj1, obj2)

    else:
        raise ValueError(
            "Both vectors must have the same format (either 'np.ndarray' or 'List[np.ndarray]')."
        )


@dataclass(frozen=True)
class TemporaryKeyword:
    term: str
    vector: t.Union[t.List[float], t.List[t.List[float]]]
    lemma: str
    pos_tag: str

    @classmethod
    def from_dict(cls, x: t.Mapping[str, t.Any]) -> TemporaryKeyword:
        return cls(x["term"], x["vector"], x["lemma"], x["pos_tag"])

    def __eq__(self, other) -> bool:
        return (
            self.term == other.term
            and self.lemma == other.lemma
            and self.pos_tag == other.pos_tag
        )

    def __hash__(self) -> int:
        return hash((self.term, self.lemma, self.pos_tag))


@dataclass(frozen=True)
class Keyword:
    term: str
    vector: Vector
    lemma: str
    pos_tag: str
    weight: float

    @classmethod
    def from_tmp(cls, obj: TemporaryKeyword, weight: float) -> Keyword:
        return cls(
            obj.term, _convert_vector(obj.vector), obj.lemma, obj.pos_tag, weight
        )


# TODO: Create keywords cache
def keywords(texts: t.Iterable[str], pos_tags: t.Iterable[str]) -> t.List[Keyword]:
    response = session.post(
        _url("keywords"),
        json={"texts": texts, "pos_tags": pos_tags},
    ).json()

    weights_map = defaultdict(list)

    for doc in response:
        for raw_keyword in doc:
            keyword = TemporaryKeyword.from_dict(raw_keyword)
            weights_map[keyword].append(raw_keyword["weight"])

    return [
        Keyword.from_tmp(k, statistics.mean(weights))
        for k, weights in weights_map.items()
    ]
