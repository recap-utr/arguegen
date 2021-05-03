from __future__ import annotations

import itertools
import statistics
import typing as t
from collections import defaultdict
from dataclasses import dataclass

import arg_services_helper
import grpc
import immutables
import nlp_service.client
import numpy as np
import requests
from arg_services.nlp.v1 import nlp_pb2, nlp_pb2_grpc
from recap_argument_graph_adaptation.controller.inflect import (
    inflect_concept,
    make_immutable,
)
from recap_argument_graph_adaptation.model.config import Config
from scipy.spatial import distance
from textacy.extract.keyterms.yake import yake

from spacy.tokens import Doc  # type: ignore

config = Config.instance()

# from sentence_transformers import SentenceTransformer

channel = grpc.insecure_channel("127.0.0.1:5001")
client = nlp_pb2_grpc.NlpServiceStub(channel)


def doc(text: str) -> Doc:
    return docs([text])[0]


def docs(texts: t.Iterable[str]) -> t.Tuple[Doc, ...]:
    docbin = client.DocBin(
        nlp_pb2.DocBinRequest(language="en", texts=texts, spacy_model="en_core_web_lg")
    ).docbin
    return nlp_service.client.docbin2doc(docbin, "en", nlp_pb2.SIMILARITY_METHOD_COSINE)


def vectors(texts: t.Iterable[str]) -> t.Tuple[np.ndarray, ...]:
    res = client.Vectors(
        nlp_pb2.VectorsRequest(
            language="en",
            texts=texts,
            spacy_model="en_core_web_lg",
            embedding_levels=[nlp_pb2.EMBEDDING_LEVEL_DOCUMENT],
        )
    )

    return tuple(nlp_service.client.list2array(x.document.vector) for x in res.vectors)


def vector(text: str) -> np.ndarray:
    return vectors([text])[0]


def similarities(text_tuples: t.Iterable[t.Tuple[str, str]]) -> t.Sequence[float]:
    return client.Similarities(
        nlp_pb2.SimilaritiesRequest(
            language="en",
            text_tuples=[
                nlp_pb2.TextTuple(text1=x[0], text2=x[1]) for x in text_tuples
            ],
            spacy_model="en_core_web_lg",
            similarity_method=nlp_pb2.SIMILARITY_METHOD_COSINE,
        )
    ).similarities


def similarity(obj1: str, obj2: str) -> float:
    return similarities([(obj1, obj2)])[0]


@dataclass(frozen=True)
class Keyword:
    keyword: str
    form2pos: immutables.Map[str, t.Tuple[str, ...]]
    pos2form: immutables.Map[str, t.Tuple[str, ...]]
    pos_tag: str
    weight: float


_keyword_cache = {}


def keywords(docs: t.Iterable[Doc], pos_tags: t.Iterable[str]) -> t.Tuple[Keyword, ...]:
    if not config.tuning("extraction", "keywords_per_adu"):
        docs = [" ".join(docs)]

    key = (tuple(doc.text for doc in docs), tuple(pos_tags))

    if key not in _keyword_cache:
        _keywords(docs, pos_tags, key)

    return _keyword_cache[key]


def _keywords(
    docs: t.Iterable[Doc],
    pos_tags: t.Iterable[str],
    key: t.Tuple[t.Tuple[str, ...], ...],
):
    keyword_map = defaultdict(list)
    keywords = []

    for doc, pos_tag in zip(docs, pos_tags):
        # https://github.com/chartbeat-labs/textacy/blob/cdedd2351bf2a56e8773ec162e08c3188809d486/src/textacy/ke/yake.py#L137
        # Textacy uses token.norm_ if normalize is None.
        # This causes for example 'centres' to be extracted as 'centers'.
        # To avoid this, we use 'lower' instead.
        for kw, score in yake(doc, include_pos=pos_tag, normalize="lower", topn=1.0):
            keyword_map[(kw, pos_tag)].append(score)

    for (kw, pos_tag), score in keyword_map.items():
        lemma, form2pos, pos2form = inflect_concept(kw, pos_tag)

        if form2pos is not None and pos2form is not None:
            keywords.append(
                Keyword(
                    keyword=lemma,
                    form2pos=form2pos,
                    pos2form=pos2form,
                    pos_tag=pos_tag,
                    weight=_dist2sim(statistics.mean(score)),
                )
            )

    keywords.sort(key=lambda x: x.weight, reverse=True)
    _keyword_cache[key] = tuple(keywords)


def _dist2sim(distance: float) -> float:
    return 1 / (1 + distance)
