import threading
import typing as t

from fastapi import FastAPI
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from pydantic import BaseModel

lock = threading.Lock()
app = FastAPI()
wn.ensure_loaded()

pos_mapping = {}


def _synset(code: str) -> Synset:
    with lock:
        return wn.synset(code)


def _synsets(name: str, pos: t.Union[str, None, t.Collection[str]]) -> t.List[Synset]:
    name = name.replace(" ", "_")

    with lock:
        results = wn.synsets(name)

    if pos:
        if isinstance(pos, str):
            pos = [pos]

        results = [ss for ss in results if str(ss.pos()) in pos]

    return results


class SynsetQuery(BaseModel):
    code: str


class SynsetPairQuery(BaseModel):
    code1: str
    code2: str


class ConceptQuery(BaseModel):
    name: str
    pos: t.Union[str, None, t.List[str]] = None


@app.get("/")
def ready() -> bool:
    return True


@app.post("/synset/definition")
def synset_definition(query: SynsetQuery) -> str:
    return _synset(query.code).definition() or ""


@app.post("/synset/examples")
def synset_examples(query: SynsetQuery) -> t.List[str]:
    return _synset(query.code).examples() or []


@app.post("/synset/hypernyms")
def synset_hypernyms(query: SynsetQuery) -> t.List[str]:
    synset = _synset(query.code)

    with lock:
        hypernyms = synset.hypernyms()

    return [h.name() for h in hypernyms if h]


@app.post("/synset/metrics")
def synset_metrics(query: SynsetPairQuery) -> t.Dict[str, t.Optional[float]]:
    s1 = _synset(query.code1)
    s2 = _synset(query.code2)

    with lock:
        return {
            "path_similarity": s1.path_similarity(s2),
            "wup_similarity": s1.wup_similarity(s2),
            "path_distance": s1.shortest_path_distance(s2),
        }


@app.post("/synsets")
def concept_synsets(query: ConceptQuery) -> t.List[str]:
    return [str(ss.name()) for ss in _synsets(query.name, query.pos) if ss]
