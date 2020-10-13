import logging
import multiprocessing
import re
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


# TODO: Multiprocessing does not work, maybe due to serialization of spacy objects.


def argument_graph(
    graph: ag.Graph,
    rule: adaptation.Rule,
    adapted_concepts: t.Mapping[Concept, Concept],
) -> None:
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
            shortest_paths_adaptations = [
                _adapt_shortest_path(*param) for param in params
            ]
        else:
            with multiprocessing.Pool() as pool:
                shortest_paths_adaptations = pool.starmap(_adapt_shortest_path, params)

        adaptation_candidates: t.Dict[Concept, int] = defaultdict(int)
        reference_length = len(all_shortest_paths[0].relationships)
        adapted_paths[root_concept] = shortest_paths_adaptations

        log.debug(
            f"Found the following candidates: {', '.join((str(path) for path in shortest_paths_adaptations))}"
        )

        for result in shortest_paths_adaptations:
            if result and len(result.relationships) == reference_length:
                name = nlp(result.end_node.processed_name)
                end_nodes = tuple([result.end_node])

                candidate = Concept(
                    name,
                    result.end_node.pos,
                    end_nodes,
                    name.similarity(rule.target.name),
                    db.distance(end_nodes, rule.target.nodes),
                )

                adaptation_candidates[candidate] += 1

        if adaptation_candidates:
            max_score = max(adaptation_candidates.values())
            most_frequent_concepts = [
                concept
                for concept, score in adaptation_candidates.items()
                if score == max_score
            ]

            adapted_concept = _filter_concepts(most_frequent_concepts, root_concept)

            # In this step, the concept is correctly capitalized.
            # Not necessary due to later grammatical correction.
            # adapted_concept = conceptnet.adapt_name(adapted_name, root_concept.name)

            adapted_concepts[root_concept] = adapted_concept

            log.info(f"Adapt ({root_concept})->({adapted_concept}).")

        else:
            log.info(f"No adaptation for ({root_concept}).")

    return adapted_concepts, adapted_paths


# TODO: Currently, only one node per concept is used in the paths. Could be improved.
# In this case, we need to update the Path class to support multiple nodes and relationships.
def _adapt_shortest_path(
    shortest_path: graph.Path,
    concept: Concept,
    rule: adaptation.Rule,
    selector: adaptation.Selector,
    method: adaptation.Method,
) -> t.Optional[graph.Path]:
    db = Database()

    # We have to convert the target to a path object here.
    start_node = (
        rule.target.best_node
        if method == adaptation.Method.WITHIN
        else concept.best_node
    )
    adapted_path = graph.Path.from_node(start_node)

    for rel in shortest_path.relationships:
        path_candidates = db.expand_nodes([adapted_path.end_node], [rel.type])

        if config["conceptnet"]["relations"]["relax_types"] and not path_candidates:
            path_candidates = db.expand_nodes([adapted_path.end_node])

        if path_candidates:
            path_candidate = _filter_paths(
                path_candidates, shortest_path, adapted_path, selector
            )

            if path_candidate:
                adapted_path = graph.Path.merge(adapted_path, path_candidate)
            else:
                return None

    return adapted_path


def _filter_concepts(
    adapted_concepts: t.Iterable[Concept], root_concept: Concept
) -> Concept:
    nlp = load.spacy_nlp()

    adapted_concepts_iter = iter(adapted_concepts)

    best_match = (next(adapted_concepts_iter), 0.0)

    for concept in adapted_concepts_iter:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = root_concept.name.similarity(concept.name)

        if sim > best_match[1]:
            best_match = (concept, sim)

    return best_match[0]


def _filter_paths(
    candidate_paths: t.Iterable[graph.Path],
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
