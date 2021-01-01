import threading
import typing as t
from collections import defaultdict, deque
from dataclasses import dataclass
from queue import Queue

from fastapi import FastAPI
from nltk.corpus import wordnet as wn
from pydantic import BaseModel

lock = threading.Lock()
app = FastAPI()
wn.ensure_loaded()


@dataclass
class Synset:
    code: str
    name: str
    pos: str


definition: t.Dict[str, str] = {}
examples: t.Dict[str, t.List[str]] = {}
hypernyms: t.Dict[str, t.List[str]] = {}
lemmas: t.Dict[str, t.List[Synset]] = defaultdict(list)

for synset in wn.all_synsets():
    code = synset.name()
    name, pos, _ = code.rsplit(".", 2)

    definition[code] = synset.definition()
    examples[code] = synset.examples()
    hypernyms[code] = [hypernym.name() for hypernym in synset.hypernyms()]

    for lemma in synset.lemmas():
        lemmas[lemma.name()].append(Synset(code, name, pos))


def _path_similarity(code1: str, code2: str) -> float:
    pass


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
    return definition[query.code]


@app.post("/synset/examples")
def synset_examples(query: SynsetQuery) -> t.List[str]:
    return examples[query.code]


@app.post("/synset/hypernyms")
def synset_hypernyms(query: SynsetQuery) -> t.List[str]:
    return hypernyms[query.code]


@app.post("/synset/metrics")
def synset_metrics(query: SynsetPairQuery) -> t.Dict[str, t.Optional[float]]:
    return {
        "path_similarity": s1.path_similarity(s2),
        "wup_similarity": s1.wup_similarity(s2),
        "path_distance": s1.shortest_path_distance(s2),
    }


@app.post("/synsets")
def concept_synsets(query: ConceptQuery) -> t.List[str]:
    name = query.name.replace(" ", "_")
    pos = query.pos

    results = lemmas[name]

    if pos:
        if isinstance(pos, str):
            pos = [pos]

        results = [result for result in results if result.pos in pos]

    return [result.code for result in results]
