from __future__ import annotations

import itertools
import json
import typing as t
from collections import defaultdict
from pathlib import Path

import nltk.corpus.reader.wordnet
import wn
import wn.similarity
from nltk.corpus.reader import WordNetCorpusReader
from recap_argument_graph_adaptation.controller import spacy

from ..model import graph
from ..model.config import Config

config = Config.instance()

db = wn.Wordnet(lang=config["nlp"]["lang"], lexicon="pwn:3.1")


# TODO: Make serialization automatic or include in git
with Path("data", "wn_exceptions.json").open() as f:
    wn_exceptions = json.load(f)

with Path("data", "wn_morphy_substitutions.json").open() as f:
    wn_morphy_substitutions = json.load(f)

lemmas = defaultdict(list)

for word in db.words():
    lemmas[word.lemma()].append(word.pos)


def _morphy(form: str, pos: str, check_exceptions: bool = True):
    # from jordanbg:
    # Given an original string x
    # 1. Apply rules once to the input to get y1, y2, y3, etc.
    # 2. Return all that are in the database
    # 3. If there are no matches, keep applying rules until you either
    #    find a match or you can't go any further
    # https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html

    exceptions = wn_exceptions[pos]
    substitutions = wn_morphy_substitutions[pos]

    def apply_rules(forms):
        return [
            form[: -len(old)] + new
            for form in forms
            for old, new in substitutions
            if form.endswith(old)
        ]

    def filter_forms(forms):
        result = []
        seen = set()

        for form in forms:
            if lemma_pos_tags := lemmas.get(form):
                if pos in lemma_pos_tags:
                    if form not in seen:
                        result.append(form)
                        seen.add(form)

        return result

    # 0. Check the exception lists
    if check_exceptions:
        if form in exceptions:
            return filter_forms([form] + exceptions[form])

    # 1. Apply rules once to the input to get y1, y2, y3, etc.
    forms = apply_rules([form])

    # 2. Return all that are in the database (and check the original too)
    results = filter_forms([form] + forms)
    if results:
        return results

    # 3. If there are no matches, keep applying rules until we find a match
    while forms:
        forms = apply_rules(forms)
        results = filter_forms(forms)
        if results:
            return results

    # Return an empty list if we can't find anything
    return []


# WORDNET API


def _synset(code: str) -> wn.Synset:
    return db.synset(code)


def _synsets(
    name: str, pos: t.Union[str, None, t.Collection[str]]
) -> t.List[wn.Synset]:
    name = name.lower()
    results = []
    pos_tags = []

    if not pos:
        # pos_tags.append(None)
        pos_tags.extend(["a", "r", "n", "v"])  # ADJ_SAT should not be included
    elif isinstance(pos, str):
        pos_tags.append(pos)
    else:
        pos_tags.extend(pos)

    for pos_tag in pos_tags:
        # results.extend(db.synsets(name, pos_tag))
        morphy_names = _morphy(name, pos_tag)

        for morphy_name in morphy_names:
            results.extend(db.synsets(morphy_name, pos_tag))

    return results


def concept_synsets(name: str, pos: t.Union[None, str, graph.POS]) -> t.List[str]:
    pos_candidates = None

    if pos:
        if isinstance(pos, graph.POS):
            pos_candidates = graph.wn_pos(pos)
        else:
            pos_candidates = [pos]

    return [ss.id for ss in _synsets(name, pos_candidates) if ss]


def synset_definition(code: str) -> str:
    return _synset(code).definition() or ""


def synset_examples(code: str) -> t.List[str]:
    return _synset(code).examples() or []


def synset_hypernyms(code: str) -> t.List[str]:
    hypernyms = _synset(code).hypernyms()

    return [h.id for h in hypernyms if h]


def synset_metrics(code1: str, code2: str) -> t.Dict[str, float]:
    s1 = _synset(code1)
    s2 = _synset(code2)
    return {
        "path_similarity": wn.similarity.path(s1, s2),
        "wup_similarity": wn.similarity.wup(s1, s2),
    }


# def hypernym_trees(code: str) -> t.List[t.List[str]]:
# hypernym_trees = [[code]]
# has_hypernyms = [True]
# final_hypernym_trees = []

# while any(has_hypernyms):
#     has_hypernyms = []
#     new_hypernym_trees = []

#     for hypernym_tree in hypernym_trees:
#         if new_hypernyms := synset_hypernyms(hypernym_tree[-1]):
#             has_hypernyms.append(True)

#             for new_hypernym in new_hypernyms:
#                 new_hypernym_trees.append([*hypernym_tree, new_hypernym])
#         else:
#             has_hypernyms.append(False)
#             final_hypernym_trees.append(hypernym_tree)

#     hypernym_trees = new_hypernym_trees

# return final_hypernym_trees

# return [[synset.id for synset in tree] for tree in _synset(code).hypernym_paths()]


# DERIVED FUNCTIONS

# TODO: Multiple lemmas are returned, make selection more robust.
def resolve(code: str) -> t.Tuple[str, graph.POS]:
    synset = _synset(code)
    lemma = synset.lemmas()[0]
    pos = graph.wn_pos_mapping[synset.pos]

    return (lemma, pos)


def contextual_synsets(text: str, term: str, pos: graph.POS) -> t.Tuple[str, ...]:
    # https://github.com/nltk/nltk/blob/develop/nltk/wsd.py
    results = concept_synsets(term, pos)

    synset_tuples = []

    for synset in results:
        similarity = 0

        if synset_def := synset_definition(synset):
            similarity = spacy.similarity(text, synset_def)

        synset_tuples.append((synset, similarity))

    synset_tuples.sort(key=lambda item: item[1])

    # Check if the best result has a higher similarity than demanded.
    # If true, only include the synsets with higher similarity.
    # Otherwise, include only the best one.
    if best_synset_tuple := next(iter(synset_tuples), None):
        if best_synset_tuple[1] > config.tuning("hypernym", "min_similarity"):
            synset_tuples = filter(
                lambda x: x[1] > config.tuning("hypernym", "min_similarity"),
                synset_tuples,
            )
        else:
            synset_tuples = (best_synset_tuple,)

    return tuple([synset for synset, _ in synset_tuples])


def contextual_synset(text: str, term: str, pos: graph.POS) -> t.Optional[str]:
    synsets = contextual_synsets(text, term, pos)

    if len(synsets) > 0:
        return synsets[0]

    return None


def metrics(
    synsets1: t.Iterable[str], synsets2: t.Iterable[str]
) -> t.Dict[str, t.Optional[float]]:
    tmp_results: t.Dict[str, t.List[float]] = {
        "path_similarity": [],
        "wup_similarity": [],
    }

    for s1, s2 in itertools.product(synsets1, synsets2):
        retrieved_metrics = synset_metrics(s1, s2)

        for key, value in retrieved_metrics.items():
            if value:
                tmp_results[key].append(value)

    results: t.Dict[str, t.Optional[float]] = {key: None for key in tmp_results.keys()}

    for key, values in tmp_results.items():
        if values:
            if "distance" in key:
                results[key] = min(values)
            else:
                results[key] = max(values)

    return results


def hypernyms(code: str) -> t.Set[str]:
    result = set()

    trees = _synset(code).hypernym_paths()

    for tree in trees:
        # The first synset is the original one, the last always entity
        tree = tree[1:-1]
        # Some synsets are not relevant for generalization
        tree = filter(
            lambda s: all(
                [
                    lemma not in config["wordnet"]["hypernym_filter"]
                    for lemma in s.lemmas()
                ]
            ),
            tree,
        )

        result.update([s.id for s in tree])

    return result
