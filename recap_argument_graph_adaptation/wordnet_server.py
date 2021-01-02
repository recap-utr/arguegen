import itertools
import math
import threading
import typing as t
from collections import defaultdict, deque
from dataclasses import dataclass
from operator import itemgetter
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


def _hypernym_trees(code: str) -> t.List[t.List[str]]:
    hypernym_trees = [[code]]
    has_hypernyms = [True]
    final_hypernym_trees = []

    while any(has_hypernyms):
        has_hypernyms = []
        new_hypernym_trees = []

        for hypernym_tree in hypernym_trees:
            if new_hypernyms := hypernyms[hypernym_tree[-1]]:
                has_hypernyms.append(True)

                for new_hypernym in new_hypernyms:
                    new_hypernym_trees.append([*hypernym_tree, new_hypernym])
            else:
                has_hypernyms.append(False)
                final_hypernym_trees.append(hypernym_tree)

        hypernym_trees = new_hypernym_trees

    return final_hypernym_trees


inf = float("inf")


def _hypernym_distances(trees: t.List[t.List[str]]) -> t.Dict[str, float]:
    result = defaultdict(lambda: inf)

    for tree in trees:
        for i, hypernym in enumerate(tree):
            result[hypernym] = min(result[hypernym], i)

    return result


def _iter_hypernym_lists(code: str):
    """
    Return an iterator over ``Synset`` objects that are either proper
    hypernyms or instance of hypernyms of the synset.

    https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html
    """
    todo = [code]
    seen = set()

    while todo:
        for synset in todo:
            seen.add(synset)

        yield todo

        todo = [
            hypernym
            for synset in todo
            for hypernym in (hypernyms[synset] + instance_hypernyms[synset])
            if hypernym not in seen
        ]


needs_root: t.Dict[str, bool] = {"*ROOT*": False}
definition: t.Dict[str, str] = {"*ROOT*": ""}
examples: t.Dict[str, t.List[str]] = {"*ROOT*": []}
hypernyms: t.Dict[str, t.List[str]] = {"*ROOT*": []}
instance_hypernyms: t.Dict[str, t.List[str]] = {"*ROOT*": []}
hypernym_trees: t.Dict[str, t.List[t.List[str]]] = {"*ROOT*": []}
all_hypernyms: t.Dict[str, t.Set[str]] = {"*ROOT*": set()}
# max_depths: t.Dict[str, int] = {}
# min_depths: t.Dict[str, int] = {}
# hypernym_distances: t.Dict[str, t.Dict[str, float]] = {}
lemmas: t.Dict[str, t.List[Synset]] = defaultdict(list)

for synset in wn.all_synsets():
    code = synset.name()
    name, pos, _ = code.rsplit(".", 2)

    needs_root[code] = synset._needs_root()
    definition[code] = synset.definition()
    examples[code] = synset.examples()
    hypernyms[code] = [hypernym.name() for hypernym in synset.hypernyms()]
    instance_hypernyms[code] = [
        hypernym.name() for hypernym in synset.instance_hypernyms()
    ]

    for lemma in synset.lemmas():
        lemmas[lemma.name()].append(Synset(code, name, pos))

for code in hypernyms.keys():
    trees = _hypernym_trees(code)
    hypernym_trees[code] = trees
    all_hypernyms[code] = set(
        synset for synsets in _iter_hypernym_lists(code) for synset in synsets
    )
    # hypernym_distances[code] = _hypernym_distances(trees)


def hypernym_paths(code):
    """
    Get the path(s) from this synset to the root, where each path is a
    list of the synset nodes traversed on the way to the root.

    https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html#Synset.hypernym_paths
    """

    paths = []
    hyps = hypernyms[code] + instance_hypernyms[code]

    if len(hyps) == 0:
        paths = [[code]]

    for hypernym in hyps:
        for ancestor_list in hypernym_paths(hypernym):
            ancestor_list.append(code)
            paths.append(ancestor_list)

    return paths


# TODO: The function max_depth and min_depth could cache/precompute their results.
def max_depth(code: str):
    """
    :return: The length of the longest hypernym path from this
    synset to the root.

    https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html#Synset.max_depth
    """

    hyps = hypernyms[code] + instance_hypernyms[code]

    if not hyps:
        result = 0
    else:
        result = 1 + max(max_depth(h) for h in hyps)

    return result


def min_depth(code: str):
    """
    :return: The length of the shortest hypernym path from this
    synset to the root.

    https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html#Synset.min_depth
    """

    hyps = hypernyms[code] + instance_hypernyms[code]
    if not hyps:
        result = 0
    else:
        result = 1 + min(min_depth(h) for h in hyps)

    return result


def common_hypernyms(code1: str, code2: str):
    """
    Find all synsets that are hypernyms of this synset and the
    other synset.

    https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html#Synset.common_hypernyms
    """

    return list(all_hypernyms[code1].intersection(all_hypernyms[code2]))


def lowest_common_hypernyms(
    code1: str, code2: str, simulate_root: bool = False, use_min_depth: bool = False
):
    """
    Get a list of lowest synset(s) that both synsets have as a hypernym.
    When `use_min_depth == False` this means that the synset which appears
    as a hypernym of both `self` and `other` with the lowest maximum depth
    is returned or if there are multiple such synsets at the same depth
    they are all returned

    However, if `use_min_depth == True` then the synset(s) which has/have
    the lowest minimum depth and appear(s) in both paths is/are returned.

    By setting the use_min_depth flag to True, the behavior of NLTK2 can be
    preserved. This was changed in NLTK3 to give more accurate results in a
    small set of cases, generally with synsets concerning people. (eg:
    'chef.n.01', 'fireman.n.01', etc.)

    This method is an implementation of Ted Pedersen's "Lowest Common
    Subsumer" method from the Perl Wordnet module. It can return either
    "self" or "other" if they are a hypernym of the other.

    https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html#Synset.lowest_common_hypernyms
    """

    synsets = common_hypernyms(code1, code2)

    if simulate_root:
        fake_synset = "*ROOT*"
        synsets.append(fake_synset)

    try:
        if use_min_depth:
            _max_depth = max(min_depth(s) for s in synsets)
            unsorted_lch = [s for s in synsets if min_depth(s) == _max_depth]
        else:
            _max_depth = max(max_depth(s) for s in synsets)
            unsorted_lch = [s for s in synsets if max_depth(s) == _max_depth]

        return sorted(unsorted_lch)

    except ValueError:
        return []


def hypernym_distances(code: str, distance: int = 0, simulate_root: bool = False):
    """
    Get the path(s) from this synset to the root, counting the distance
    of each node from the initial node on the way. A set of
    (synset, distance) tuples is returned.

    https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html#Synset.hypernym_distances
    """

    distances = set([(code, distance)])

    for hypernym in hypernyms[code] + instance_hypernyms[code]:
        distances |= hypernym_distances(hypernym, distance + 1, simulate_root=False)

    if simulate_root:
        fake_synset = "*ROOT*"
        fake_synset_distance = max(distances, key=itemgetter(1))[1]
        distances.add((fake_synset, fake_synset_distance + 1))

    return distances


def _shortest_hypernym_paths(code: str, simulate_root: bool):
    # https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html

    if code == "*ROOT*":
        return {code: 0}

    queue = deque([(code, 0)])
    path = {}

    while queue:
        s, depth = queue.popleft()
        if s in path:
            continue
        path[s] = depth

        depth += 1
        queue.extend((hyp, depth) for hyp in hypernyms[s])  # type: ignore
        queue.extend((hyp, depth) for hyp in instance_hypernyms[s])  # type: ignore

    if simulate_root:
        fake_synset = "*ROOT*"
        path[fake_synset] = max(path.values()) + 1

    return path


def shortest_path_distance(
    code1: str, code2: str, simulate_root=False
) -> t.Optional[float]:
    """
    Returns the distance of the shortest path linking the two synsets (if
    one exists). For each synset, all the ancestor nodes and their
    distances are recorded and compared. The ancestor node common to both
    synsets that can be reached with the minimum number of traversals is
    used. If no ancestor nodes are common, None is returned. If a node is
    compared with itself 0 is returned.

    https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html#Synset.shortest_path_distance
    """

    if code1 == code2:
        return 0

    # dist_dict1 = hypernym_distances[code1]
    # dist_dict2 = hypernym_distances[code2]
    dist_dict1 = _shortest_hypernym_paths(code1, simulate_root)
    dist_dict2 = _shortest_hypernym_paths(code2, simulate_root)

    inf = float("inf")
    path_distance = inf

    for synset, d1 in dist_dict1.items():
        d2 = dist_dict2.get(synset, inf)
        path_distance = min(path_distance, d1 + d2)

    return None if math.isinf(path_distance) else path_distance


def path_similarity(code1: str, code2: str, simulate_root: bool = True):
    """
    Path Distance Similarity:
    Return a score denoting how similar two word senses are, based on the
    shortest path that connects the senses in the is-a (hypernym/hypnoym)
    taxonomy. The score is in the range 0 to 1, except in those cases where
    a path cannot be found (will only be true for verbs as there are many
    distinct verb taxonomies), in which case None is returned. A score of
    1 represents identity i.e. comparing a sense with itself will return 1.

    https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html#Synset.path_similarity
    """

    distance = shortest_path_distance(
        code1, code2, simulate_root and (needs_root[code1] or needs_root[code2])
    )
    if distance is None or distance < 0:
        return None

    return 1.0 / (distance + 1)


def wup_similarity(code1: str, code2: str, simulate_root: bool = True):
    """
    Wu-Palmer Similarity:
    Return a score denoting how similar two word senses are, based on the
    depth of the two senses in the taxonomy and that of their Least Common
    Subsumer (most specific ancestor node). Previously, the scores computed
    by this implementation did _not_ always agree with those given by
    Pedersen's Perl implementation of WordNet Similarity. However, with
    the addition of the simulate_root flag (see below), the score for
    verbs now almost always agree but not always for nouns.

    The LCS does not necessarily feature in the shortest path connecting
    the two senses, as it is by definition the common ancestor deepest in
    the taxonomy, not closest to the two senses. Typically, however, it
    will so feature. Where multiple candidates for the LCS exist, that
    whose shortest path to the root node is the longest will be selected.
    Where the LCS has multiple paths to the root, the longer path is used
    for the purposes of the calculation.

    https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html#Synset.wup_similarity
    """

    _needs_root = needs_root[code1] or needs_root[code2]
    # Note that to preserve behavior from NLTK2 we set use_min_depth=True
    # It is possible that more accurate results could be obtained by
    # removing this setting and it should be tested later on
    subsumers = lowest_common_hypernyms(
        code1, code2, simulate_root=simulate_root and _needs_root, use_min_depth=True
    )

    # If no LCS was found return None
    if len(subsumers) == 0:
        return None

    subsumer = code if code in subsumers else subsumers[0]

    # Get the longest path from the LCS to the root,
    # including a correction:
    # - add one because the calculations include both the start and end
    #   nodes
    depth = max_depth(subsumer) + 1

    # Note: No need for an additional add-one correction for non-nouns
    # to account for an imaginary root node because that is now
    # automatically handled by simulate_root
    # if subsumer._pos != NOUN:
    #     depth += 1

    # Get the shortest path from the LCS to each of the synsets it is
    # subsuming.  Add this to the LCS path length to get the path
    # length from each synset to the root.
    len1 = shortest_path_distance(
        code1, subsumer, simulate_root=simulate_root and _needs_root
    )
    len2 = shortest_path_distance(
        code2, subsumer, simulate_root=simulate_root and _needs_root
    )

    if len1 is None or len2 is None:
        return None

    len1 += depth
    len2 += depth

    return (2.0 * depth) / (len1 + len2)


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


@app.post("/synset/hypernyms/trees")
def synset_hypernym_trees(query: SynsetQuery) -> t.List[t.List[str]]:
    return hypernym_trees[query.code]


@app.post("/synset/metrics")
def synset_metrics(query: SynsetPairQuery) -> t.Dict[str, t.Optional[float]]:
    s1 = query.code1
    s2 = query.code2

    return {
        "path_similarity": path_similarity(s1, s2),
        "wup_similarity": wup_similarity(s1, s2),
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
