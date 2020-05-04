import logging
import multiprocessing
import typing as t
from collections import defaultdict
import warnings

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


def argument_graph(
    graph: ag.Graph, rule: t.Tuple[str, str], adapted_concepts: t.Dict[Concept, str]
) -> None:
    for node in graph.inodes:
        # First perform the rule adaptation.
        # Otherwise, 'death penalty' might already be replaced using the adaptation of 'death'.
        node.text = node.text.replace(rule[0], rule[1])

        for concept, adapted_concept in adapted_concepts.items():
            node.text = node.text.replace(concept.original_name, adapted_concept,)


def paths(
    reference_paths: t.Mapping[Concept, t.Sequence[graph.Path]],
    rule: t.Tuple[str, str],
    selector: adaptation.Selector,
    method: adaptation.Method,
) -> t.Tuple[t.Dict[Concept, str], t.Dict[Concept, t.List[graph.Path]]]:
    adapted_concepts = {}
    adapted_paths = {}

    for root_concept, all_shortest_paths in reference_paths.items():
        log.debug(f"Adapting '{root_concept}'.")

        params = [
            (shortest_path, root_concept, rule, selector, method)
            for shortest_path in all_shortest_paths
        ]

        with multiprocessing.Pool() as pool:
            shortest_paths_adaptations = pool.starmap(_adapt_shortest_path, params)

        adaptation_candidates = defaultdict(int)
        reference_length = len(all_shortest_paths[0].relationships)
        adapted_paths[root_concept] = shortest_paths_adaptations

        log.debug(
            f"Found the following candidates: {', '.join((str(path) for path in shortest_paths_adaptations))}"
        )

        for result in shortest_paths_adaptations:
            if result and len(result.relationships) == reference_length:
                adaptation_candidates[result.end_node.processed_name] += 1

        if adaptation_candidates:
            max_value = max(adaptation_candidates.values())
            adapted_names = [
                key
                for key, value in adaptation_candidates.items()
                if value == max_value
            ]
            adapted_name = _filter_concepts(adapted_names, root_concept)
            adapted_name = conceptnet.adapt_name(
                adapted_name, root_concept.original_name
            )

            adapted_concepts[root_concept] = adapted_name

            log.info(f"Adapt ({root_concept})->({adapted_name}).")

        else:
            log.info(f"No adaptation for ({root_concept}).")

    return adapted_concepts, adapted_paths


def _adapt_shortest_path(
    shortest_path: graph.Path,
    concept: Concept,
    rule: t.Tuple[str, str],
    selector: adaptation.Selector,
    method: adaptation.Method,
) -> t.Optional[graph.Path]:
    db = Database()

    # We have to convert the target to a path object here.
    start_name = (
        rule[1] if method == adaptation.Method.WITHIN else concept.conceptnet_name
    )
    adapted_path = graph.Path.from_node(db.node(start_name))

    for rel in shortest_path.relationships:
        path_candidates = db.expand_node(adapted_path.end_node, [rel.type])

        if config["adaptation"]["relax_relationship_types"] and not path_candidates:
            path_candidates = db.expand_node(adapted_path.end_node)

        path_candidate = _filter_paths(
            path_candidates, shortest_path, adapted_path, selector
        )

        if path_candidate:
            adapted_path = graph.Path.merge(adapted_path, path_candidate)
        else:
            return None

    return adapted_path


def _filter_concepts(adapted_concepts: t.Iterable[str], root_concept: Concept) -> str:
    nlp = load.spacy_nlp()

    root_nlp = nlp(root_concept.conceptnet_name)
    adapted_concepts_iter = iter(adapted_concepts)

    best_match = (next(adapted_concepts_iter), 0.0)

    for concept in adapted_concepts_iter:
        concept_nlp = nlp(concept)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = root_nlp.similarity(concept_nlp)

        if sim > best_match[1]:
            best_match = (concept, sim)

    return best_match[0]


def _filter_paths(
    candidate_paths: t.Optional[t.Iterable[graph.Path]],
    reference_path: graph.Path,
    adapted_path: graph.Path,
    selector: adaptation.Selector,
) -> t.Optional[graph.Path]:
    nlp = load.spacy_nlp()

    end_index = len(adapted_path.nodes)
    start_index = end_index - 1

    val_reference = _aggregate_features(
        nlp(reference_path.nodes[start_index].processed_name).vector,
        nlp(reference_path.nodes[end_index].processed_name).vector,
        selector,
    )

    solution_pair = (None, 1.0)
    existing_nodes = set(adapted_path.nodes)

    for candidate_path in candidate_paths:
        candidate = candidate_path.end_node

        # if candidate not in existing_nodes:
        val_adapted = _aggregate_features(
            nlp(adapted_path.nodes[start_index].processed_name).vector,
            nlp(candidate.processed_name).vector,
            selector,
        )
        val_solution = _compare_features(val_reference, val_adapted, selector)

        if val_solution < solution_pair[1]:
            solution_pair = (candidate_path, val_solution)

    return solution_pair[0]


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
