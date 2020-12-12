import statistics
import logging
import multiprocessing
import re

from nltk.corpus.reader.wordnet import Synset
from recap_argument_graph_adaptation.controller import metrics, wordnet
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
# TODO: Adaption is langsam, wahrscheinlich weil die Knotendistanz von ConceptNet berechnet wird.

related_concept_weight = config["nlp"]["related_concept_weight"]

if round(sum(related_concept_weight.values()), 2) != 1:
    raise ValueError("The sum is not 1.")


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
        # node.text = pr.proofread(node.text)


def _replace(input_text: str, substitutions: t.Mapping[str, str]) -> str:
    """Perform multiple replacements in a single run."""

    substrings = sorted(substitutions.keys(), key=len, reverse=True)
    regex = re.compile("|".join(map(re.escape, substrings)))  # type: ignore

    return regex.sub(lambda match: substitutions[match.group(0)], input_text)


def synsets(
    concepts: t.Iterable[Concept], rule: adaptation.Rule
) -> t.Tuple[t.Dict[Concept, Concept], t.Dict[Concept, t.Set[Concept]]]:
    adapted_synsets = {}
    adapted_concepts = {}
    nlp = load.spacy_nlp()

    for original_concept in concepts:
        adaptation_candidates = set()

        for synset in original_concept.synsets:
            hypernyms = wordnet.hypernyms(synset)

            for hypernym in hypernyms:
                _name, pos = wordnet.resolve_synset(hypernym)
                name = nlp(_name)
                nodes = tuple()
                synsets = (hypernym,)

                related_concepts = {
                    rule.target: related_concept_weight["rule_target"],
                    rule.source: related_concept_weight["rule_source"],
                    original_concept: related_concept_weight["original_concept"],
                }

                candidate = Concept(
                    name,
                    pos,
                    nodes,
                    synsets,
                    None,
                    *metrics.init_concept_metrics(
                        name, nodes, synsets, related_concepts
                    ),
                )

                adaptation_candidates.add(candidate)

        adapted_synsets[original_concept] = adaptation_candidates
        adapted_concept = _filter_concepts(adaptation_candidates, rule)

        if adapted_concept:
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

    for original_concept, all_shortest_paths in reference_paths.items():
        log.debug(f"Adapting '{original_concept}'.")

        params = [
            (shortest_path, original_concept, rule, selector, method)
            for shortest_path in all_shortest_paths
        ]

        if config["debug"]:
            adaptation_results = [_adapt_shortest_path(*param) for param in params]
        else:
            with multiprocessing.Pool() as pool:
                adaptation_results = pool.starmap(_adapt_shortest_path, params)

        adaptation_results = list(itertools.chain(*adaptation_results))
        adapted_paths[original_concept] = adaptation_results
        adaptation_candidates = set()
        log.debug(
            f"Found the following candidates: {', '.join((str(path) for path in adaptation_results))}"
        )

        for result in adaptation_results:
            name = nlp(result.end_node.processed_name)
            end_nodes = tuple([result.end_node])
            pos = result.end_node.pos
            synsets = wordnet.synsets(name.text, pos)
            related_concepts = {
                rule.target: related_concept_weight["rule_target"],
                rule.source: related_concept_weight["rule_source"],
                original_concept: related_concept_weight["original_concept"],
            }

            candidate = Concept(
                name,
                pos,
                end_nodes,
                synsets,
                None,
                *metrics.init_concept_metrics(
                    name, end_nodes, synsets, related_concepts
                ),
            )

            adaptation_candidates.add(candidate)

        adapted_concept = _filter_concepts(adaptation_candidates, rule)

        if adapted_concept:
            # In this step, the concept is correctly capitalized.
            # Not necessary due to later grammatical correction.
            # adapted_concept = conceptnet.adapt_name(adapted_name, root_concept.name)

            adapted_concepts[original_concept] = adapted_concept
            log.info(f"Adapt ({original_concept})->({adapted_concept}).")

        else:
            log.info(f"No adaptation for ({original_concept}).")

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
    concepts: t.Set[Concept], rule: adaptation.Rule
) -> t.Optional[Concept]:
    # Remove the original adaptation source from the candidates
    filtered_concepts = concepts.difference([rule.source])
    filtered_concepts = Concept.only_relevant(
        filtered_concepts, config["nlp"]["min_score_adaptation"]
    )

    if filtered_concepts:
        sorted_concepts = sorted(
            filtered_concepts,
            key=lambda c: c.score,
            reverse=True,
        )

        return sorted_concepts[0]

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

    return sorted_candidates[: config["conceptnet"]["bfs_node_limit"]]


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
