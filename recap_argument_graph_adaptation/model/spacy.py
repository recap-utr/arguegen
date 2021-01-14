import typing as t

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


def vector(text: str) -> np.ndarray:
    # return _nlp(text).vector

    if text not in _vector_cache:
        _vector_cache[text] = np.array(
            session.post(_url("vector"), json={"text": text}).json()
        )

    return _vector_cache[text]


# This function does not use the cache!
def vectors(texts: t.Iterable[str]) -> t.List[np.ndarray]:
    unknown_texts = []

    for text in texts:
        if text not in _vector_cache:
            unknown_texts.append(text)

    if unknown_texts:
        vectors = session.post(_url("vectors"), json={"texts": unknown_texts}).json()

        for text, vector in zip(unknown_texts, vectors):
            _vector_cache[text] = np.array(vector)

    return [_vector_cache[text] for text in texts]


def similarity(obj1: t.Union[str, np.ndarray], obj2: t.Union[str, np.ndarray]) -> float:
    if isinstance(obj1, str):
        obj1 = vector(obj1)

    if isinstance(obj2, str):
        obj2 = vector(obj2)

    if np.any(obj1) and np.any(obj2):
        return float(1 - distance.cosine(obj1, obj2))

    return 0.0


def keywords(
    texts: t.Iterable[str], pos_tags: t.Iterable[str]
) -> t.List[t.Dict[str, t.Any]]:
    response = session.post(
        _url("keywords"),
        json={"texts": texts, "pos_tags": pos_tags},
    ).json()

    for doc in response:
        doc["vector"] = np.array(doc["vector"])

        for keyword in doc["keywords"]:
            keyword["vector"] = np.array(keyword["vector"])

    return response
