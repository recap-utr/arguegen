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

_cache = {}

channel = grpc.insecure_channel("127.0.0.1:5001")
client = nlp_pb2_grpc.NlpServiceStub(channel)


def vectors(texts: t.Iterable[str]) -> t.Tuple[np.ndarray, ...]:
    unknown = {text for text in texts if text not in _cache}

    if unknown:
        res = client.Vectors(
            nlp_pb2.VectorsRequest(
                language="en",
                texts=texts,
                spacy_model="en_core_web_lg",
                embedding_levels=[nlp_pb2.EMBEDDING_LEVEL_DOCUMENT],
            )
        )

        new_vectors = tuple(
            nlp_service.client.list2array(x.document.vector) for x in res.vectors
        )
        _cache.update({text: vec for text, vec in zip(unknown, new_vectors)})

    return tuple(_cache[text] for text in texts)


def vector(text: str) -> np.ndarray:
    return vectors([text])[0]


def similarities(text_tuples: t.Iterable[t.Tuple[str, str]]) -> t.Tuple[float, ...]:
    vecs1 = vectors([x[0] for x in text_tuples])
    vecs2 = vectors([x[1] for x in text_tuples])

    return tuple(1 - distance.cosine(vec1, vec2) for vec1, vec2 in zip(vecs1, vecs2))


def similarity(text1: str, text2: str) -> float:
    vecs = vectors([text1, text2])

    return 1 - distance.cosine(vecs[0], vecs[1])


@dataclass(frozen=True)
class Keyword:
    keyword: str
    form2pos: immutables.Map[str, t.Tuple[str, ...]]
    pos2form: immutables.Map[str, t.Tuple[str, ...]]
    pos_tag: str
    weight: float


_keyword_cache = {}


def keywords(
    texts: t.Iterable[str], pos_tags: t.Iterable[str]
) -> t.Tuple[Keyword, ...]:
    if not config.tuning("extraction", "keywords_per_adu"):
        texts = [" ".join(texts)]

    key = (tuple(texts), tuple(pos_tags))

    if key not in _keyword_cache:
        _keyword_cache[key] = tuple(_keywords(texts, pos_tags))

    return _keyword_cache[key]


def _keywords(
    texts: t.Iterable[str],
    pos_tags: t.Iterable[str],
) -> t.List[Keyword]:
    docbin = client.DocBin(
        nlp_pb2.DocBinRequest(
            language="en",
            texts=texts,
            spacy_model="en_core_web_lg",
        )
    ).docbin
    docs = nlp_service.client.docbin2doc(docbin, "en", nlp_pb2.SIMILARITY_METHOD_COSINE)
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

    return keywords


def _dist2sim(distance: float) -> float:
    return 1 / (1 + distance)
