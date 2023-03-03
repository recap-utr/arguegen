from __future__ import annotations

import inspect
import statistics
import typing as t
from collections import defaultdict
from dataclasses import dataclass

import grpc
import immutables
import nlp_service.client
import nlp_service.similarity
import nlp_service.typing
import numpy as np
import numpy.typing as npt
import spacy
from arg_services.nlp.v1 import nlp_pb2, nlp_pb2_grpc
from spacy.language import Language
from spacy.tokens import Doc
from textacy.extract.keyterms.yake import yake

from arguegen.controllers.inflect import inflect_concept

Vector = nlp_service.typing.NumpyVector

_Item = t.TypeVar("_Item")


def _split_list(
    seq: t.Sequence[_Item],
) -> t.Tuple[t.Sequence[_Item], t.Sequence[_Item]]:
    half = len(seq) // 2
    return seq[:half], seq[half:]


def _flatten_list(seq: t.Iterable[t.Tuple[_Item, _Item]]) -> t.List[_Item]:
    return [item[0] for item in seq] + [item[1] for item in seq]


def _dist2sim(distance: float) -> float:
    return 1 / (1 + distance)


@dataclass(frozen=True)
class Keyword:
    lemma: str
    form2pos: immutables.Map[str, t.Tuple[str, ...]]
    pos2form: immutables.Map[str, t.Tuple[str, ...]]
    pos_tag: str
    weight: float


KeywordKey = tuple[tuple[str, ...], tuple[str, ...]]


class Nlp:
    _client: nlp_pb2_grpc.NlpServiceStub
    _config: nlp_pb2.NlpConfig
    _blank_model: Language
    _vector_cache: dict[str, Vector] = {}
    _keyword_cache: dict[KeywordKey, tuple[Keyword, ...]] = {}
    _doc_cache: dict[str, Doc] = {}

    def __init__(
        self,
        address: str,
        config: nlp_pb2.NlpConfig,
    ):
        channel = grpc.insecure_channel(
            address, [("grpc.lb_policy_name", "round_robin")]
        )
        self._client = nlp_pb2_grpc.NlpServiceStub(channel)
        self._config = config
        self._blank_model = spacy.blank(config.language)

    def vectors(self, texts: t.Iterable[str]) -> t.Tuple[np.ndarray, ...]:
        if unknown_texts := [text for text in texts if text not in self._vector_cache]:
            res = self._client.Vectors(
                nlp_pb2.VectorsRequest(
                    texts=unknown_texts,
                    embedding_levels=[nlp_pb2.EMBEDDING_LEVEL_DOCUMENT],
                    config=self._config,
                )
            )

            new_vectors = tuple(np.array(x.document.vector) for x in res.vectors)
            self._vector_cache.update(zip(unknown_texts, new_vectors))

        return tuple(self._vector_cache[text] for text in texts)

    def vector(self, text: str) -> npt.NDArray[np.float_]:
        return self.vectors([text])[0]

    def similarities(
        self, text_tuples: t.Iterable[t.Tuple[str, str]]
    ) -> t.Tuple[float, ...]:
        if inspect.isgenerator(text_tuples):
            text_tuples = list(text_tuples)

        vecs = self.vectors(_flatten_list(text_tuples))
        vecs1, vecs2 = _split_list(vecs)

        return tuple(
            nlp_service.similarity.cosine(vec1, vec2)
            for vec1, vec2 in zip(vecs1, vecs2)
        )

    def similarity(self, text1: str, text2: str) -> float:
        return self.similarities([(text1, text2)])[0]

    def parse_docs(
        self,
        texts: t.Iterable[str],
        attributes: t.Optional[t.Iterable[str]] = None,
    ) -> t.Tuple[Doc, ...]:
        attrs = nlp_pb2.Strings(values=attributes) if attributes is not None else None

        if unknown_texts := [text for text in texts if text not in self._doc_cache]:
            req = nlp_pb2.DocBinRequest(
                texts=unknown_texts,
                config=nlp_pb2.NlpConfig(
                    language=self._config.language, spacy_model="en_core_web_sm"
                ),
                attributes=attrs,
            )
            docbin = self._client.DocBin(req).docbin
            docs = nlp_service.client.docbin2docs(docbin, self._blank_model)
            self._doc_cache.update(zip(unknown_texts, docs))

        return tuple(self._doc_cache[text] for text in texts)

    def parse_doc(
        self, text: str, attributes: t.Optional[t.Iterable[str]] = None
    ) -> Doc:
        return self.parse_docs([text], attributes)[0]

    def keywords(
        self, texts: t.Iterable[str], pos_tags: t.Iterable[str], per_adu: bool = False
    ) -> t.Tuple[Keyword, ...]:
        if not per_adu:
            texts = [" ".join(texts)]

        key = (tuple(texts), tuple(pos_tags))

        if key not in self._keyword_cache:
            self._keyword_cache[key] = tuple(self._keywords(texts, pos_tags))

        return self._keyword_cache[key]

    def _keywords(
        self,
        texts: t.Iterable[str],
        pos_tags: t.Iterable[str],
    ) -> t.List[Keyword]:
        docs = self.parse_docs(texts)
        keyword_map = defaultdict(list)
        keywords = []

        for doc, pos_tag in zip(docs, pos_tags):
            for kw, score in yake(
                doc, include_pos=pos_tag, normalize="lemma", topn=1.0
            ):
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
