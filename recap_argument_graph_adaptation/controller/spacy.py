import typing as t

import numpy as np
import requests
from recap_argument_graph_adaptation.model.config import config
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer


spacy_cache = {}
proof_reader_cache = {}
spacy_models = {
    "de-integrated": "de_core_news_lg",
    "de-transformer": "de_core_news_sm",
    "en-integrated": "en_core_web_lg",
    "en-transformer": "en_core_web_sm",
}
transformer_models = {
    "de": "distiluse-base-multilingual-cased",
    "en": "roberta-large-nli-stsb-mean-tokens",
}

_base_url = f"http://{config['spacy']['host']}:{config['spacy']['port']}"
session = requests.Session()


def _url(*parts: str) -> str:
    return "/".join([_base_url, *parts])


# def _nlp() -> Language:
#     lang = config["nlp"]["lang"]
#     embeddings = config["nlp"]["embeddings"]
#     model_name = f"{lang}-{embeddings}"

#     if not spacy_cache.get(model_name):
#         model = spacy.load(
#             spacy_models[model_name], disable=["ner", "textcat", "parser"]
#         )  # parser needed for noun chunks
#         model.add_pipe(model.create_pipe("sentencizer"))

#         if embeddings == "transformer":
#             model.add_pipe(TransformerModel(lang), first=True)

#         spacy_cache[model_name] = model

#     return spacy_cache[model_name]  # type: ignore


# def _proof_reader() -> lmproof.Proofreader:
#     lang = config["nlp"]["lang"]

#     if not proof_reader_cache.get(lang):
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             proof_reader_cache[lang] = lmproof.load(lang)  # type: ignore

#     return proof_reader_cache[lang]  # type: ignore


# https://spacy.io/usage/processing-pipelines#custom-components-user-hooks
# https://github.com/explosion/spaCy/issues/3823
class TransformerModel(object):
    def __init__(self, lang):
        self._model = SentenceTransformer(transformer_models[lang])

    def __call__(self, doc):
        doc.user_hooks["vector"] = self.vector
        doc.user_span_hooks["vector"] = self.vector
        doc.user_token_hooks["vector"] = self.vector

        doc.user_hooks["similarity"] = self.similarity
        doc.user_span_hooks["similarity"] = self.similarity
        doc.user_token_hooks["similarity"] = self.similarity

        return doc

    def vector(self, obj):
        # The function `encode` expects a list of strings.
        sentences = [obj.text]
        embeddings = self._model.encode(sentences)

        return embeddings[0]

    def similarity(self, obj1, obj2):
        if np.any(obj1) and np.any(obj2):
            return 1 - distance.cosine(obj1.vector, obj2.vector)

        return 0.0


def vector(text: str) -> np.ndarray:
    # return _nlp(text).vector
    return np.array(session.post(_url("vector"), json={"text": text}).json())


def similarity(obj1: t.Union[str, np.ndarray], obj2: t.Union[str, np.ndarray]) -> float:
    if isinstance(obj1, str) and isinstance(obj2, str):
        # return _nlp(obj1).similarity(_nlp(obj2))
        return float(
            session.post(_url("similarity"), json={"text1": obj1, "text2": obj2}).json()
        )

    if isinstance(obj1, str):
        obj1 = vector(obj1)

    if isinstance(obj2, str):
        obj2 = vector(obj2)

    if np.any(obj1) and np.any(obj2):
        return float(1 - distance.cosine(obj1, obj2))

    return 0.0


def keywords(
    text: str, pos_tags: t.Iterable[str]
) -> t.List[t.Tuple[str, str, str, float]]:
    # if not pos_tags:
    #     pos_tags = []

    # doc = _nlp(text)
    # terms = []
    # normalize_func = "lemma" if normalize else None

    # for pos_tag in pos_tags:
    #     keywords = extractor(doc, include_pos=pos_tag, normalize=normalize_func)
    #     terms.extend(((keyword, pos_tag, weight) for keyword, weight in keywords))

    # return terms

    return session.post(
        _url("keywords"),
        json={"text": text, "pos_tags": pos_tags},
    ).json()
