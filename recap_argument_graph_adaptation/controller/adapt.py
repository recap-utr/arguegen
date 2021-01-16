import itertools
import logging
import re
import typing as t

import recap_argument_graph as ag
from recap_argument_graph_adaptation.model import casebase, graph, query, spacy
from recap_argument_graph_adaptation.model.config import Config

config = Config.instance()

log = logging.getLogger(__name__)


def argument_graph(
    original_graph: ag.Graph,
    rules: t.Collection[casebase.Rule],
    adapted_concepts: t.Mapping[casebase.Concept, casebase.Concept],
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


def concepts(
    concepts: t.Iterable[casebase.Concept], rules: t.Collection[casebase.Rule]
) -> t.Tuple[
    t.Dict[casebase.Concept, casebase.Concept],
    t.Dict[casebase.Concept, t.Set[casebase.Concept]],
]:
    all_candidates = {}
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

        for node in original_concept.nodes:
            hypernym_distances = node.hypernym_distances()

            for hypernym, hyp_distance in hypernym_distances.items():
                name = hypernym.processed_name
                pos = hypernym.pos
                vector = spacy.vector(name)
                nodes = frozenset([hypernym])

                candidate = casebase.Concept(
                    name,
                    vector,
                    query.pos(pos),
                    nodes,
                    query.concept_metrics(
                        nodes, vector, None, hyp_distance, related_concepts
                    ),
                )

                adaptation_candidates.add(candidate)

        all_candidates[original_concept] = adaptation_candidates
        adapted_concept = _filter_concepts(
            adaptation_candidates, original_concept, rules
        )

        if adapted_concept:
            adapted_concepts[original_concept] = adapted_concept
            log.debug(f"Adapt ({original_concept})->({adapted_concept}).")

        else:
            log.debug(f"No adaptation for ({original_concept}).")

    return adapted_concepts, all_candidates


def paths(
    reference_paths: t.Mapping[casebase.Concept, t.Sequence[graph.AbstractPath]],
    rules: t.Collection[casebase.Rule],
) -> t.Tuple[
    t.Dict[casebase.Concept, casebase.Concept],
    t.Dict[casebase.Concept, t.List[graph.AbstractPath]],
]:
    related_concept_weight = config.tuning("weight")
    adapted_concepts = {}
    adapted_paths = {}

    for original_concept, all_shortest_paths in reference_paths.items():
        adaptation_results = [
            _bfs_adaptation(shortest_path, original_concept, rule)
            for shortest_path, rule in itertools.product(all_shortest_paths, rules)
        ]

        adaptation_results = list(itertools.chain(*adaptation_results))
        adapted_paths[original_concept] = adaptation_results
        adaptation_candidates = set()

        for result in adaptation_results:
            hyp_distance = len(result)
            name = result.end_node.processed_name
            vector = spacy.vector(name)
            end_nodes = frozenset([result.end_node])
            pos = query.pos(result.end_node.pos)
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

            candidate = casebase.Concept(
                name,
                vector,
                pos,
                end_nodes,
                query.concept_metrics(
                    end_nodes, vector, None, hyp_distance, related_concepts
                ),
            )

            adaptation_candidates.add(candidate)

        adapted_concept = _filter_concepts(
            adaptation_candidates, original_concept, rules
        )

        if adapted_concept:
            adapted_concepts[original_concept] = adapted_concept
            log.debug(f"Adapt ({original_concept})->({adapted_concept}).")

        else:
            log.debug(f"No adaptation for ({original_concept}).")

    return adapted_concepts, adapted_paths


def _bfs_adaptation(
    shortest_path: graph.AbstractPath,
    concept: casebase.Concept,
    rule: casebase.Rule,
) -> t.List[graph.AbstractPath]:
    method = config.tuning("bfs", "method")
    adapted_paths = []

    # We have to convert the target to a path object here.
    start_nodes = rule.target.nodes if method == "within" else concept.nodes

    for start_node in start_nodes:
        current_paths = []
        current_paths.append(
            graph.AbstractPath.from_node(start_node)
        )  # Start with only one node.

        for _ in shortest_path.relationships:
            next_paths = []

            for current_path in current_paths:
                path_extensions = query.hypernyms_as_paths(current_path.end_node)

                if path_extensions:
                    path_extensions = _filter_paths(
                        current_path, path_extensions, shortest_path
                    )

                    for path_extension in path_extensions:
                        next_paths.append(
                            graph.AbstractPath.merge(current_path, path_extension)
                        )

            current_paths = next_paths

        adapted_paths.extend(current_paths)

    return adapted_paths


def _filter_concepts(
    concepts: t.Set[casebase.Concept],
    original_concept: casebase.Concept,
    rules: t.Collection[casebase.Rule],
) -> t.Optional[casebase.Concept]:
    # Remove the original adaptation source from the candidates
    filter_expr = [rule.source for rule in rules] + [original_concept]
    filtered_concepts = concepts.difference(filter_expr)
    filtered_concepts = casebase.filter_concepts(
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


# TODO: Incorporate the hypernym level s.t. adaptations with a similar level of generalization are preferred.
def _filter_paths(
    current_path: graph.AbstractPath,
    path_extensions: t.Iterable[graph.AbstractPath],
    reference_path: graph.AbstractPath,
) -> t.List[graph.AbstractPath]:
    selector = config.tuning("bfs", "selector")
    candidate_values = {}

    start_index = len(current_path.relationships)
    end_index = start_index + 1

    val_reference = _aggregate_features(
        spacy.vector(reference_path.nodes[start_index].processed_name),
        spacy.vector(reference_path.nodes[end_index].processed_name),
        selector,
    )

    for candidate in path_extensions:
        val_adapted = _aggregate_features(
            spacy.vector(current_path.end_node.processed_name),
            spacy.vector(candidate.end_node.processed_name),
            selector,
        )
        candidate_values[candidate] = _compare_features(
            val_reference, val_adapted, selector
        )

    sorted_candidate_tuples = sorted(
        candidate_values.items(), key=lambda x: x[1], reverse=True
    )
    sorted_candidates = [x[0] for x in sorted_candidate_tuples]

    return sorted_candidates[: config["adaptation"]["bfs_node_limit"]]


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
