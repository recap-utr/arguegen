import itertools
import logging
import re
import typing as t

import recap_argument_graph as ag
from recap_argument_graph_adaptation.controller import metrics, spacy, wordnet

from ..model import adaptation, graph
from ..model.adaptation import Concept
from ..model.config import config
from ..model.database import Database

log = logging.getLogger(__name__)


def argument_graph(
    original_graph: ag.Graph,
    rules: t.Collection[adaptation.Rule],
    adapted_concepts: t.Mapping[Concept, Concept],
) -> ag.Graph:
    substitutions = {
        concept.name: adapted_concept.name
        for concept, adapted_concept in adapted_concepts.items()
    }
    for rule in rules:
        substitutions[rule.source.name] = rule.target.name

    adapted_graph = original_graph.copy()

    for node in adapted_graph.inodes:
        node.text = _replace(node.plain_text, substitutions)
        # node.text = pr.proofread(node.text)

    return adapted_graph


def _replace(text: str, substitutions: t.Mapping[str, str]):
    substrings = sorted(substitutions.keys(), key=len, reverse=True)

    for substring in substrings:
        pattern = re.compile(substring, re.IGNORECASE)
        orig = pattern.sub(substitutions[substring], text)

    return text


def synsets(
    concepts: t.Iterable[Concept], rules: t.Collection[adaptation.Rule]
) -> t.Tuple[t.Dict[Concept, Concept], t.Dict[Concept, t.Set[Concept]]]:
    adapted_synsets = {}
    adapted_concepts = {}
    related_concept_weight = config.tuning("weight")

    for original_concept in concepts:
        adaptation_candidates = set()
        related_concepts = {}

        for rule in rules:
            related_concepts.update(
                {
                    rule.target: related_concept_weight["rule_target"] / len(rules),
                    rule.source: related_concept_weight["rule_source"] / len(rules),
                    original_concept: related_concept_weight["original_concept"]
                    / len(rules),
                }
            )

        for synset in original_concept.synsets:
            hypernyms = wordnet.hypernyms(synset)

            for hypernym in hypernyms:
                name, pos = wordnet.resolve(hypernym)
                vector = spacy.vector(name)
                nodes = tuple()
                synsets = (hypernym,)

                candidate = Concept(
                    name,
                    vector,
                    pos,
                    nodes,
                    synsets,
                    None,
                    *metrics.init_concept_metrics(
                        vector, nodes, synsets, related_concepts
                    ),
                )

                adaptation_candidates.add(candidate)

        adapted_synsets[original_concept] = adaptation_candidates
        adapted_concept = _filter_concepts(
            adaptation_candidates, original_concept, rules
        )

        if adapted_concept:
            adapted_concepts[original_concept] = adapted_concept
            log.debug(f"Adapt ({original_concept})->({adapted_concept}).")

        else:
            log.debug(f"No adaptation for ({original_concept}).")

    return adapted_concepts, adapted_synsets


def paths(
    reference_paths: t.Mapping[Concept, t.Sequence[graph.Path]],
    rules: t.Collection[adaptation.Rule],
) -> t.Tuple[t.Dict[Concept, Concept], t.Dict[Concept, t.List[graph.Path]]]:
    related_concept_weight = config.tuning("weight")
    adapted_concepts = {}
    adapted_paths = {}

    for original_concept, all_shortest_paths in reference_paths.items():
        adaptation_results = [
            _adapt_shortest_path(shortest_path, original_concept, rule)
            for shortest_path, rule in itertools.product(all_shortest_paths, rules)
        ]

        adaptation_results = list(itertools.chain(*adaptation_results))
        adapted_paths[original_concept] = adaptation_results
        adaptation_candidates = set()

        for result in adaptation_results:
            name = result.end_node.processed_name
            vector = spacy.vector(name)
            end_nodes = tuple([result.end_node])
            pos = result.end_node.pos
            synsets = wordnet.concept_synsets(name, pos)
            related_concepts = {}

            for rule in rules:
                related_concepts.update(
                    {
                        rule.target: related_concept_weight["rule_target"] / len(rules),
                        rule.source: related_concept_weight["rule_source"] / len(rules),
                        original_concept: related_concept_weight["original_concept"]
                        / len(rules),
                    }
                )

            candidate = Concept(
                name,
                vector,
                pos,
                end_nodes,
                tuple(synsets),
                None,
                *metrics.init_concept_metrics(
                    vector, end_nodes, synsets, related_concepts
                ),
            )

            adaptation_candidates.add(candidate)

        adapted_concept = _filter_concepts(
            adaptation_candidates, original_concept, rules
        )

        if adapted_concept:
            # In this step, the concept is correctly capitalized.
            # Not necessary due to later grammatical correction.
            # adapted_concept = conceptnet.adapt_name(adapted_name, root_concept.name)

            adapted_concepts[original_concept] = adapted_concept
            log.debug(f"Adapt ({original_concept})->({adapted_concept}).")

        else:
            log.debug(f"No adaptation for ({original_concept}).")

    return adapted_concepts, adapted_paths


def _adapt_shortest_path(
    shortest_path: graph.Path,
    concept: Concept,
    rule: adaptation.Rule,
) -> t.List[graph.Path]:
    db = Database()
    method = config.tuning("conceptnet", "method")
    current_paths = []

    # We have to convert the target to a path object here.
    start_node = rule.target.best_node if method == "within" else concept.best_node
    current_paths.append(graph.Path.from_node(start_node))  # Start with only one node.

    for rel in shortest_path.relationships:
        next_paths = []

        for current_path in current_paths:
            path_candidates = db.expand_nodes([current_path.end_node], [rel.type])

            if config["conceptnet"]["relation"]["relax_types"] and not path_candidates:
                path_candidates = db.expand_nodes([current_path.end_node])

            if path_candidates:
                path_candidates = _filter_paths(
                    path_candidates, shortest_path, start_node
                )

                for path_candidate in path_candidates:
                    next_paths.append(graph.Path.merge(current_path, path_candidate))

        current_paths = next_paths

    return current_paths


def _filter_concepts(
    concepts: t.Set[Concept],
    original_concept: Concept,
    rules: t.Collection[adaptation.Rule],
) -> t.Optional[Concept]:
    # Remove the original adaptation source from the candidates
    filter_expr = [rule.source for rule in rules] + [original_concept]
    filtered_concepts = concepts.difference(filter_expr)
    filtered_concepts = Concept.only_relevant(
        filtered_concepts, config.tuning("adaptation", "min_score")
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
) -> t.List[graph.Path]:
    selector = config.tuning("conceptnet", "selector")
    candidate_values = {}

    end_index = len(candidate_paths[0].nodes) - 1
    start_index = end_index - 1

    val_reference = _aggregate_features(
        spacy.vector(reference_path.nodes[start_index].processed_name),
        spacy.vector(reference_path.nodes[end_index].processed_name),
        selector,
    )

    for candidate_path in candidate_paths:
        candidate = candidate_path.end_node

        val_adapted = _aggregate_features(
            spacy.vector(start_node.processed_name),
            spacy.vector(candidate.processed_name),
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


def _aggregate_features(feat1: t.Any, feat2: t.Any, selector: str) -> t.Any:
    if selector == "difference":
        return abs(feat1 - feat2)
    elif selector == "similarity":
        return spacy.similarity(feat1, feat2)

    raise ValueError("Parameter 'selector' wrong.")


def _compare_features(feat1: t.Any, feat2: t.Any, selector: str) -> t.Any:
    if selector == "difference":
        return spacy.similarity(feat1, feat2)
    elif selector == "similarity":
        return abs(feat1 - feat2)

    raise ValueError("Parameter 'selector' wrong.")
