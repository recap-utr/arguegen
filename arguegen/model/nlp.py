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
import spacy
from arg_services.nlp.v1 import nlp_pb2, nlp_pb2_grpc
from arguegen.controller.inflect import inflect_concept, make_immutable
from arguegen.model.config import Config
from scipy.spatial import distance
from spacy.tokens import Doc, DocBin
from textacy.extract.keyterms.yake import yake

config = Config.instance()

_vector_cache = {}


def init_client():
    channel = grpc.insecure_channel(
        config["resources"]["nlp"]["url"], [("grpc.lb_policy_name", "round_robin")]
    )
    return nlp_pb2_grpc.NlpServiceStub(channel)


client = init_client()

_vector_config = {
    "glove": nlp_pb2.NlpConfig(
        language=config["nlp"]["lang"],
        spacy_model="en_core_web_lg",
    ),
    "use": nlp_pb2.NlpConfig(
        language=config["nlp"]["lang"],
        embedding_models=[
            nlp_pb2.EmbeddingModel(
                model_type=nlp_pb2.EMBEDDING_TYPE_USE,
                model_name="https://tfhub.dev/google/universal-sentence-encoder/4",
                pooling=nlp_pb2.POOLING_MEAN,
            )
        ],
    ),
    # "use-large": {
    #     "embedding_models": [
    #         nlp_pb2.EmbeddingModel(
    #             model_type=nlp_pb2.EMBEDDING_TYPE_USE,
    #             model_name="https://tfhub.dev/google/universal-sentence-encoder-large/5",
    #             pooling=nlp_pb2.POOLING_MEAN,
    #         )
    #     ],
    # },
    "sbert": nlp_pb2.NlpConfig(
        language=config["nlp"]["lang"],
        embedding_models=[
            nlp_pb2.EmbeddingModel(
                model_type=nlp_pb2.EMBEDDING_TYPE_SBERT,
                model_name="stsb-mpnet-base-v2",
                pooling=nlp_pb2.POOLING_MEAN,
            )
        ],
    ),
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


@dataclass(frozen=True)
class TextKey:
    text: str
    use_token_vectors: t.Optional[bool] = None
    embeddings: t.Optional[str] = None


def vectors(texts: t.Iterable[str]) -> t.Tuple[np.ndarray, ...]:
    args = (use_token_vectors(), config.tuning("nlp", "embeddings"))
    text_keys = [TextKey(text, *args) for text in texts]
    unknown_keys = [key for key in text_keys if key not in _vector_cache]

    if unknown_keys:
        if use_token_vectors():
            levels = [nlp_pb2.EMBEDDING_LEVEL_TOKENS]
        else:
            levels = [nlp_pb2.EMBEDDING_LEVEL_DOCUMENT]

        res = client.Vectors(
            nlp_pb2.VectorsRequest(
                texts=[x.text for x in unknown_keys],
                embedding_levels=levels,
                config=_vector_config[config.tuning("nlp", "embeddings")],
            )
        )

        if use_token_vectors():
            new_vectors = tuple(
                tuple(np.array(token.vector) for token in x.tokens) for x in res.vectors
            )
        else:
            new_vectors = tuple(np.array(x.document.vector) for x in res.vectors)

        _vector_cache.update({x: vec for x, vec in zip(unknown_keys, new_vectors)})

    return tuple(_vector_cache[key] for key in text_keys)


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


_doc_cache = {}
_nlp = spacy.blank(config["nlp"]["lang"])


def parse_docs(
    texts: t.Iterable[str],
    attributes: t.Iterable[str] = tuple(),
    pipes: t.Iterable[str] = tuple(),
) -> t.Tuple[Doc, ...]:
    text_keys = [TextKey(text) for text in texts]
    unknown_keys = [key for key in text_keys if key not in _vector_cache]

    if unknown_keys:
        docbin = client.DocBin(
            nlp_pb2.DocBinRequest(
                texts=[key.text for key in unknown_keys],
                config=nlp_pb2.NlpConfig(
                    language=config["nlp"]["lang"], spacy_model="en_core_web_sm"
                ),
                attributes=attributes,
                pipes=pipes,
            )
        ).docbin
        docs = nlp_service.client.docbin2docs(docbin, _nlp)
        _doc_cache.update({key: doc for key, doc in zip(unknown_keys, docs)})

    return tuple(_doc_cache[key] for key in text_keys)


def parse_doc(
    text: str, attributes: t.Iterable[str] = tuple(), pipes: t.Iterable[str] = tuple()
) -> Doc:
    return parse_docs([text], attributes)[0]


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
    if config.tuning_run and not config.tuning("extraction", "keywords_per_adu"):
        texts = [" ".join(texts)]

    key = (tuple(texts), tuple(pos_tags))

    if key not in _keyword_cache:
        _keyword_cache[key] = tuple(_keywords(texts, pos_tags))

    return _keyword_cache[key]


def _keywords(
    texts: t.Iterable[str],
    pos_tags: t.Iterable[str],
) -> t.List[Keyword]:
    docs = parse_docs(texts)
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
