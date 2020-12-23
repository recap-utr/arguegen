import typing as t
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

wn.ensure_loaded()


def _synset(code: str) -> Synset:
    return wn.synset(code)


def _synsets(name: str, pos: t.Optional[str]) -> t.List[Synset]:
    results = wn.synsets(name.replace(" ", "_"))

    if pos:
        results = [ss for ss in results if str(ss.pos()) == pos]

    return results


def _plain_synsets(name: str, pos: t.Optional[str]) -> t.List[str]:
    return [ss.name() for ss in _synsets(name, pos) if ss]  # type: ignore


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ("/RPC2",)


with SimpleXMLRPCServer(
    ("localhost", 8000), requestHandler=RequestHandler, allow_none=True
) as server:
    server.register_introspection_functions()

    # Register a function under function.__name__.
    @server.register_function()
    def synset_definition(code: str) -> str:
        return _synset(code).definition() or ""

    @server.register_function()
    def synset_examples(code: str) -> t.List[str]:
        return _synset(code).examples() or []

    @server.register_function()
    def synset_hypernyms(code: str) -> t.List[str]:
        hypernyms = _synset(code).hypernyms()
        return [h.name() for h in hypernyms if h]

    @server.register_function()
    def synset_metrics(code1: str, code2: str) -> t.Dict[str, t.Optional[float]]:
        s1 = _synset(code1)
        s2 = _synset(code2)

        return {
            "path_similarity": s1.path_similarity(s2),
            "wup_similarity": s1.wup_similarity(s2),
            "path_distance": s1.shortest_path_distance(s2),
        }

    @server.register_function()
    def concept_synsets(name: str, pos: t.Optional[str] = None) -> t.List[str]:
        return _plain_synsets(name, pos)

    server.serve_forever()
