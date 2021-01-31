from __future__ import annotations

import json
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


def _check_response(response: requests.Response) -> None:
    if not response.ok:
        raise RuntimeError(response.content)


def _url(*parts: str) -> str:
    return "/".join([_base_url, *parts])


_vector_cache = {}
Vector = t.Union[np.ndarray, t.Tuple[np.ndarray, ...]]


def _convert_vector(
    vector: t.Union[t.Tuple[t.Tuple[float, ...], ...], t.Tuple[float, ...]]
) -> Vector:
    if any(isinstance(i, float) for i in vector):
        return np.array(vector)
    else:
        return tuple(np.array(v) for v in vector)


def vector(text: str) -> Vector:
    if text not in _vector_cache:
        response = session.post(_url("vector"), json={"text": text})
        _check_response(response)

        _vector_cache[text] = _convert_vector(response.json())

    return _vector_cache[text]


# This function does not use the cache!
def vectors(texts: t.Iterable[str]) -> t.Tuple[Vector]:
    unknown_texts = []

    for text in texts:
        if text not in _vector_cache:
            unknown_texts.append(text)

    if unknown_texts:
        response = session.post(_url("vectors"), json={"texts": unknown_texts})
        _check_response(response)

        for text, vector in zip(unknown_texts, response.json()):
            _vector_cache[text] = _convert_vector(vector)

    return tuple(_vector_cache[text] for text in texts)


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
    obj1: t.Union[str, Vector],
    obj2: t.Union[str, Vector],
) -> float:
    if isinstance(obj1, str):
        obj1 = vector(obj1)
    if isinstance(obj2, str):
        obj2 = vector(obj2)

    if isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
        if np.any(obj1) and np.any(obj2):
            return float(1 - distance.cosine(obj1, obj2))
        return 0.0

    elif isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
        return dynamax_jaccard(obj1, obj2)

    else:
        raise ValueError(
            "Both vectors must have the same format (either 'np.ndarray' or 'List[np.ndarray]')."
        )


def inflect(keyword: str, pos_tags: t.Iterable[t.Optional[str]]) -> t.Dict[str, t.Any]:
    response = session.post(
        _url("inflect"),
        json={"keyword": keyword, "pos_tags": pos_tags},
    )
    _check_response(response)

    result = response.json()
    result["vector"] = _convert_vector(result["vector"])
    result["forms"] = frozenset(result["forms"])

    return result


@dataclass(frozen=True)
class TemporaryKeyword:
    keyword: str
    vector: t.Union[t.Tuple[float, ...], t.Tuple[t.Tuple[float, ...], ...]]
    forms: t.FrozenSet[str]
    pos_tag: str

    @classmethod
    def from_dict(cls, x: t.Mapping[str, t.Any]) -> TemporaryKeyword:
        return cls(x["keyword"], x["vector"], x["forms"], x["pos_tag"])

    def __eq__(self, other) -> bool:
        return (
            self.keyword == other.keyword
            and self.forms == other.forms
            and self.pos_tag == other.pos_tag
        )

    def __hash__(self) -> int:
        return hash((self.keyword, self.forms, self.pos_tag))


@dataclass(frozen=True)
class Keyword:
    keyword: str
    vector: Vector
    forms: t.FrozenSet[str]
    pos_tag: str
    weight: float

    @classmethod
    def from_tmp(cls, obj: TemporaryKeyword, weight: float) -> Keyword:
        return cls(
            obj.keyword,
            _convert_vector(obj.vector),
            obj.forms,
            obj.pos_tag,
            weight,
        )


def keywords(texts: t.Iterable[str], pos_tags: t.Iterable[str]) -> t.List[Keyword]:
    weights_map = defaultdict(list)
    response = session.post(
        _url("keywords"),
        json={
            "texts": texts,
            "pos_tags": pos_tags,
        },
    )
    _check_response(response)

    for doc in response.json():
        for raw_keyword in doc:
            keyword = TemporaryKeyword.from_dict(raw_keyword)
            weights_map[keyword].append(raw_keyword["weight"])

    candidates = [
        Keyword.from_tmp(k, statistics.mean(weights))
        for k, weights in weights_map.items()
    ]
    candidates.sort(key=lambda x: x.weight, reverse=True)

    return candidates
