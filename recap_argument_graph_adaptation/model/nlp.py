from __future__ import annotations

import inspect
import itertools
import statistics
import typing as t
from collections import defaultdict
from dataclasses import dataclass

import arg_services_helper
import grpc
import immutables
import nlp_service.client
import nlp_service.similarity
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

config = Config.instance()

_vector_cache = {}


def init_client():
    global client
    global channel

    channel = grpc.insecure_channel(
        config["resources"]["nlp"]["url"], [("grpc.lb_policy_name", "round_robin")]
    )
    client = nlp_pb2_grpc.NlpServiceStub(channel)


client = None
channel = None

_vector_config = {
    "glove": {
        "spacy_model": "en_core_web_lg",
    },
    "use-small": {
        "embedding_models": [
            nlp_pb2.EmbeddingModel(
                model_type=nlp_pb2.EMBEDDING_TYPE_USE,
                model_name="https://tfhub.dev/google/universal-sentence-encoder/4",
                pooling=nlp_pb2.POOLING_MEAN,
            )
        ],
    },
    "use-large": {
        "embedding_models": [
            nlp_pb2.EmbeddingModel(
                model_type=nlp_pb2.EMBEDDING_TYPE_USE,
                model_name="https://tfhub.dev/google/universal-sentence-encoder-large/5",
                pooling=nlp_pb2.POOLING_MEAN,
            )
        ],
    },
    "sbert": {
        "embedding_models": [
            nlp_pb2.EmbeddingModel(
                model_type=nlp_pb2.EMBEDDING_TYPE_SBERT,
                model_name="roberta-large-nli-stsb-mean-tokens",
                pooling=nlp_pb2.POOLING_MEAN,
            )
        ],
    },
}

_similarity_config = {
    "cosine": nlp_service.similarity.cosine,
    "dynamax": nlp_service.similarity.dynamax_jaccard,
}

use_token_vectors = lambda: config.tuning("nlp", "similarity") in ["dynamax"]


def _split_list(
    seq: t.Sequence[t.Any],
) -> t.Tuple[t.Sequence[t.Any], t.Sequence[t.Any]]:
    half = len(seq) // 2
    return seq[:half], seq[half:]


def _flatten_list(seq: t.Iterable[t.Tuple[t.Any, t.Any]]) -> t.List[t.Any]:
    return [item[0] for item in seq] + [item[1] for item in seq]


def vectors(texts: t.Iterable[str]) -> t.Tuple[np.ndarray, ...]:
    if inspect.isgenerator(texts):
        texts = list(texts)

    unknown = {text for text in texts if text not in _vector_cache}

    if unknown:
        if use_token_vectors():
            levels = [nlp_pb2.EMBEDDING_LEVEL_TOKENS]
        else:
            levels = [nlp_pb2.EMBEDDING_LEVEL_DOCUMENT]

        res = client.Vectors(
            nlp_pb2.VectorsRequest(
                language=config["nlp"]["lang"],
                texts=texts,
                embedding_levels=levels,
                **_vector_config[config.tuning("nlp", "embeddings")]
            )
        )

        if use_token_vectors():
            new_vectors = tuple(
                tuple(nlp_service.client.list2array(token.vector) for token in x.tokens)
                for x in res.vectors
            )
        else:
            new_vectors = tuple(
                nlp_service.client.list2array(x.document.vector) for x in res.vectors
            )

        _vector_cache.update({text: vec for text, vec in zip(unknown, new_vectors)})

    return tuple(_vector_cache[text] for text in texts)


def vector(text: str) -> np.ndarray:
    return vectors([text])[0]


def similarities(text_tuples: t.Iterable[t.Tuple[str, str]]) -> t.Tuple[float, ...]:
    if inspect.isgenerator(text_tuples):
        text_tuples = list(text_tuples)

    vecs = vectors(_flatten_list(text_tuples))
    vecs1, vecs2 = _split_list(vecs)

    return tuple(
        _similarity_config[config.tuning("nlp", "similarity")](vec1, vec2)
        for vec1, vec2 in zip(vecs1, vecs2)
    )


def similarity(text1: str, text2: str) -> float:
    return similarities([(text1, text2)])[0]


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
            language=config["nlp"]["lang"],
            texts=texts,
            spacy_model="en_core_web_lg",
        )
    ).docbin
    docs = nlp_service.client.docbin2doc(docbin, config["nlp"]["lang"])
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
