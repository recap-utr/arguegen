import logging
import multiprocessing
import re

from nltk.corpus.reader.wordnet import Synset
from recap_argument_graph_adaptation.controller import wordnet
import typing as t
from collections import defaultdict
import warnings
import itertools

import recap_argument_graph as ag
import spacy
from scipy.spatial import distance

from . import load
from ..model import graph, adaptation
from ..util import conceptnet
from ..model.adaptation import Concept
from ..model.config import config
from ..model.database import Database

log = logging.getLogger(__name__)


# TODO: Multiprocessing does not work, maybe due to serialization of spacy objects.


def argument_graph(
    graph: ag.Graph,
    rule: adaptation.Rule,
    adapted_concepts: t.Mapping[Concept, Concept],
) -> None:
    # TODO: Was passiert mit Konzepten, die ein anderes beinhalten (also bspw. school uniforms und uniforms)
    pr = load.proof_reader()
    substitutions = {
        concept.name.text: adapted_concept.name.text
        for concept, adapted_concept in adapted_concepts.items()
    }
    substitutions[rule.source.name.text] = rule.target.name.text

    for node in graph.inodes:
        node.text = _replace(node.text, substitutions)
        node.text = pr.proofread(node.text)


def _replace(input_text: str, substitutions: t.Mapping[str, str]) -> str:
    """Perform multiple replacements in a single run."""

    substrings = sorted(substitutions.keys(), key=len, reverse=True)
    regex = re.compile("|".join(map(re.escape, substrings)))

    return regex.sub(lambda match: substitutions[match.group(0)], input_text)


def synsets(
    concepts: t.Iterable[Concept], rule: adaptation.Rule
) -> t.Tuple[t.Dict[Concept, Concept], t.Dict[Concept, t.Set[Synset]]]:
    adapted_synsets = {}
    adapted_concepts = {}
    nlp = load.spacy_nlp()

    for original_concept in concepts:
        adaptation_results = set()

        for synset in original_concept.synsets:
            adaptation_results.update(wordnet.hypernyms(synset))

        adapted_synsets[original_concept] = adaptation_results
        adaptation_candidates: t.Dict[Concept, int] = defaultdict(int)

        for result in adaptation_results:
            _name, pos = wordnet.resolve_synset(result)
            name = nlp(_name)
            nodes = tuple()
            synsets = (result,)

            candidate = Concept(
                name,
                pos,
                nodes,
                synsets,
                name.similarity(rule.target.name),
                config["nlp"]["max_distance"],
                *wordnet.metrics(synsets, rule.target.synsets),
            )

            adaptation_candidates[candidate] += 1

        adapted_concept = _filter_concepts(adaptation_candidates, rule)

        if adapted_concept:
            # In this step, the concept is correctly capitalized.
            # Not necessary due to later grammatical correction.
            # adapted_concept = conceptnet.adapt_name(adapted_name, root_concept.name)

            adapted_concepts[original_concept] = adapted_concept
            log.info(f"Adapt ({original_concept})->({adapted_concept}).")

        else:
            log.info(f"No adaptation for ({original_concept}).")

    return adapted_concepts, adapted_synsets


def paths(
    reference_paths: t.Mapping[Concept, t.Sequence[graph.Path]],
    rule: adaptation.Rule,
    selector: adaptation.Selector,
    method: adaptation.Method,
) -> t.Tuple[t.Dict[Concept, Concept], t.Dict[Concept, t.List[graph.Path]]]:
    nlp = load.spacy_nlp()
    db = Database()

    adapted_concepts = {}
    adapted_paths = {}

    for root_concept, all_shortest_paths in reference_paths.items():
        log.debug(f"Adapting '{root_concept}'.")

        params = [
            (shortest_path, root_concept, rule, selector, method)
            for shortest_path in all_shortest_paths
        ]

        if config["debug"]:
            adaptation_results = [_adapt_shortest_path(*param) for param in params]
        else:
            with multiprocessing.Pool() as pool:
                adaptation_results = pool.starmap(_adapt_shortest_path, params)

        adaptation_results = list(itertools.chain(*adaptation_results))
        adapted_paths[root_concept] = adaptation_results
        adaptation_candidates: t.Dict[Concept, int] = defaultdict(int)

        log.debug(
            f"Found the following candidates: {', '.join((str(path) for path in adaptation_results))}"
        )

        for result in adaptation_results:
            name = nlp(result.end_node.processed_name)
            end_nodes = tuple([result.end_node])
            pos = result.end_node.pos
            synsets = wordnet.synsets(name.text, pos)

            candidate = Concept(
                name,
                pos,
                end_nodes,
                synsets,
                name.similarity(rule.target.name),
                db.distance(end_nodes, rule.target.nodes),
                *wordnet.metrics(synsets, rule.target.synsets),
            )

            adaptation_candidates[candidate] += 1

        adapted_concept = _filter_concepts(adaptation_candidates, rule)

        if adapted_concept:
            # In this step, the concept is correctly capitalized.
            # Not necessary due to later grammatical correction.
            # adapted_concept = conceptnet.adapt_name(adapted_name, root_concept.name)

            adapted_concepts[root_concept] = adapted_concept
            log.info(f"Adapt ({root_concept})->({adapted_concept}).")

        else:
            log.info(f"No adaptation for ({root_concept}).")

    return adapted_concepts, adapted_paths


def _adapt_shortest_path(
    shortest_path: graph.Path,
    concept: Concept,
    rule: adaptation.Rule,
    selector: adaptation.Selector,
    method: adaptation.Method,
) -> t.List[graph.Path]:
    db = Database()

    # We have to convert the target to a path object here.
    start_node = (
        rule.target.best_node
        if method == adaptation.Method.WITHIN
        else concept.best_node
    )
    current_paths = [graph.Path.from_node(start_node)]  # Start with only one node.

    for rel in shortest_path.relationships:
        next_paths = []

        for current_path in current_paths:
            path_candidates = db.expand_nodes([current_path.end_node], [rel.type])

            if config["conceptnet"]["relation"]["relax_types"] and not path_candidates:
                path_candidates = db.expand_nodes([current_path.end_node])

            if path_candidates:
                path_candidates = _filter_paths(
                    path_candidates, shortest_path, start_node, selector
                )

                for path_candidate in path_candidates:
                    next_paths.append(graph.Path.merge(current_path, path_candidate))

        current_paths = next_paths

    return current_paths


def _filter_concepts(
    concept_occurrences: t.Mapping[Concept, int], rule: adaptation.Rule
) -> t.Optional[Concept]:
    # Remove the original adaptation source from the candidated
    filtered_concepts = set(concept_occurrences).difference([rule.source])

    # Sort key: occurrences * similarity * 1/distance
    score = lambda concept: (
        concept.wordnet_score
        if config["adaptation"]["knowledge_graph"] == "wordnet"
        else concept.conceptnet_score
    )

    # TODO: Anderes Verfahren finden, um ein Konzept auszuwählen.
    # Momentan wird oft whole genommen, obwohl bei "house" bspw. "building" oder "structure" besser wären.

    if filtered_concepts:
        sorted_concepts = sorted(
            filtered_concepts,
            key=lambda concept: concept_occurrences[concept] * score(concept),
            reverse=True,
        )

        candidate = sorted_concepts[0]

        if score(candidate) >= 0.1:
            return candidate

    return None


def _filter_paths(
    candidate_paths: t.Sequence[graph.Path],
    reference_path: graph.Path,
    start_node: graph.Node,
    selector: adaptation.Selector,
) -> t.List[graph.Path]:
    nlp = load.spacy_nlp()
    candidate_values = {}

    end_index = len(candidate_paths[0].nodes) - 1
    start_index = end_index - 1

    val_reference = _aggregate_features(
        nlp(reference_path.nodes[start_index].processed_name).vector,
        nlp(reference_path.nodes[end_index].processed_name).vector,
        selector,
    )

    for candidate_path in candidate_paths:
        candidate = candidate_path.end_node

        val_adapted = _aggregate_features(
            nlp(start_node.processed_name).vector,
            nlp(candidate.processed_name).vector,
            selector,
        )
        candidate_values[candidate_path] = _compare_features(
            val_reference, val_adapted, selector
        )

    sorted_candidate_tuples = sorted(
        candidate_values.items(), key=lambda x: x[1], reverse=True
    )
    sorted_candidates = [x[0] for x in sorted_candidate_tuples]

    return sorted_candidates[: config["adaptation"]["bfs_node_limit"]]


def _aggregate_features(
    feat1: t.Any, feat2: t.Any, selector: adaptation.Selector
) -> t.Any:
    if selector == adaptation.Selector.DIFFERENCE:
        return abs(feat1 - feat2)
    elif selector == adaptation.Selector.SIMILARITY:
        return _cosine(feat1, feat2)

    raise ValueError("Parameter 'selector' wrong.")


def _compare_features(
    feat1: t.Any, feat2: t.Any, selector: adaptation.Selector
) -> t.Any:
    if selector == adaptation.Selector.DIFFERENCE:
        return _cosine(feat1, feat2)
    elif selector == adaptation.Selector.SIMILARITY:
        return abs(feat1 - feat2)

    raise ValueError("Parameter 'selector' wrong.")


def _cosine(feat1, feat2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return distance.cosine(feat1, feat2)
