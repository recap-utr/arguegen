from __future__ import annotations

import inspect
import itertools
import statistics
import typing as t
from collections import defaultdict
from dataclasses import dataclass

import grpc
import immutables
import nlp_service.client
import nlp_service.similarity
import numpy as np
import numpy.typing as npt
import requests
import spacy
from arg_services.nlp.v1 import nlp_pb2, nlp_pb2_grpc
from scipy.spatial import distance
from spacy.tokens import Doc, DocBin
from textacy.extract.keyterms.yake import yake

from arguegen.config import config, tuning, tuning_run
from arguegen.controller.inflect import inflect_concept, make_immutable

_vector_cache = {}


def init_client():
    channel = grpc.insecure_channel(
        config["resources"]["nlp"]["url"], [("grpc.lb_policy_name", "round_robin")]
    )
    return nlp_pb2_grpc.NlpServiceStub(channel)


client = init_client()

_vector_config = {
    "glove": nlp_pb2.NlpConfig(
        language=config.nlp.lang,
        spacy_model="en_core_web_lg",
    ),
    "use": nlp_pb2.NlpConfig(
        language=config.nlp.lang,
        embedding_models=[
            nlp_pb2.EmbeddingModel(
                model_type=nlp_pb2.EMBEDDING_TYPE_TENSORFLOW_HUB,
                model_name="https://tfhub.dev/google/universal-sentence-encoder/4",
                pooling_type=nlp_pb2.POOLING_MEAN,
            )
        ],
    ),
    "sbert": nlp_pb2.NlpConfig(
        language=config.nlp.lang,
        embedding_models=[
            nlp_pb2.EmbeddingModel(
                model_type=nlp_pb2.EMBEDDING_TYPE_SENTENCE_TRANSFORMERS,
                model_name="stsb-mpnet-base-v2",
                pooling_type=nlp_pb2.POOLING_MEAN,
            )
        ],
    ),
}


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
    embeddings: t.Optional[str] = None


def vectors(texts: t.Iterable[str]) -> t.Tuple[np.ndarray, ...]:
    text_keys = [TextKey(text, tuning(config, "nlp", "embeddings")) for text in texts]

    if unknown_keys := [key for key in text_keys if key not in _vector_cache]:
        res = client.Vectors(
            nlp_pb2.VectorsRequest(
                texts=[x.text for x in unknown_keys],
                embedding_levels=[nlp_pb2.EMBEDDING_LEVEL_DOCUMENT],
                config=_vector_config[tuning(config, "nlp", "embeddings")],
            )
        )

        new_vectors = tuple(np.array(x.document.vector) for x in res.vectors)
        _vector_cache.update(dict(zip(unknown_keys, new_vectors)))

    return tuple(_vector_cache[key] for key in text_keys)


def vector(text: str) -> npt.NDArray[np.float_]:
    return vectors([text])[0]


def similarities(text_tuples: t.Iterable[t.Tuple[str, str]]) -> t.Tuple[float, ...]:
    if inspect.isgenerator(text_tuples):
        text_tuples = list(text_tuples)

    vecs = vectors(_flatten_list(text_tuples))
    vecs1, vecs2 = _split_list(vecs)

    return tuple(
        nlp_service.similarity.cosine(vec1, vec2) for vec1, vec2 in zip(vecs1, vecs2)
    )


def similarity(text1: str, text2: str) -> float:
    return similarities([(text1, text2)])[0]


_doc_cache = {}
_nlp = spacy.blank(config.nlp.lang)


def parse_docs(
    texts: t.Iterable[str],
    attributes: t.Optional[t.Iterable[str]] = None,
) -> t.Tuple[Doc, ...]:
    text_keys = [TextKey(text) for text in texts]
    attrs = nlp_pb2.Strings(values=attributes) if attributes is not None else None

    if unknown_keys := [key for key in text_keys if key not in _vector_cache]:
        req = nlp_pb2.DocBinRequest(
            texts=[key.text for key in unknown_keys],
            config=nlp_pb2.NlpConfig(
                language=config.nlp.lang, spacy_model="en_core_web_sm"
            ),
            attributes=attrs,
        )
        docbin = client.DocBin(req).docbin
        docs = nlp_service.client.docbin2docs(docbin, _nlp)
        _doc_cache.update(dict(zip(unknown_keys, docs)))

    return tuple(_doc_cache[key] for key in text_keys)


def parse_doc(text: str, attributes: t.Optional[t.Iterable[str]] = None) -> Doc:
    return parse_docs([text], attributes)[0]


@dataclass(frozen=True)
class Keyword:
    lemma: str
    form2pos: immutables.Map[str, t.Tuple[str, ...]]
    pos2form: immutables.Map[str, t.Tuple[str, ...]]
    pos_tag: str
    weight: float


_keyword_cache = {}


def keywords(
    texts: t.Iterable[str], pos_tags: t.Iterable[str]
) -> t.Tuple[Keyword, ...]:
    if tuning_run(config) and not tuning(config, "extraction", "keywords_per_adu"):
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
        for kw, score in yake(doc, include_pos=pos_tag, normalize="lemma", topn=1.0):
            keyword_map[(kw, pos_tag)].append(score)

    for (kw, pos_tag), score in keyword_map.items():
        lemma, form2pos, pos2form = inflect_concept(kw, pos_tag, lemmatize=False)

        if form2pos is not None and pos2form is not None:
            keywords.append(
                Keyword(
                    lemma=lemma,
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
