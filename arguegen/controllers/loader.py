import contextlib
import itertools
import logging
import re
import typing as t

import arguebuf as ag
from arg_services.cbr.v1beta import adaptation_pb2, model_pb2

from arguegen.config import LoaderConfig
from arguegen.controllers.inflect import inflect_concept
from arguegen.model import casebase, wordnet
from arguegen.model.nlp import Nlp

log = logging.getLogger(__name__)


# TODO: May be worth investigting for performance improvements
class Loader:
    def __init__(
        self,
        case_name: str,
        case: model_pb2.AnnotatedGraph,
        query: model_pb2.AnnotatedGraph,
        nlp: Nlp,
        wn: wordnet.Wordnet,
        config: LoaderConfig,
    ):
        self.case_name = case_name
        self.case_graph = casebase.Graph.load(case)
        self.query_graph = casebase.Graph.load(query)
        self.query = query
        self.nlp = nlp
        self.wn = wn
        self.config = config

    def parse(
        self,
        case_request: adaptation_pb2.AdaptedCaseRequest,
    ) -> casebase.Case:
        rules = (
            self._parse_rules(case_request.rules)
            if case_request.rules
            else self._generate_rules()
        )

        return casebase.Case(self.case_name, self.query_graph, self.case_graph, rules)

    def _parse_rules(
        self, rules: t.Iterable[adaptation_pb2.Rule]
    ) -> t.Tuple[casebase.Rule, ...]:
        result = []

        for rule in rules:
            source = self._parse_concept(rule.source, None)
            target = self._parse_concept(rule.target, source.atoms)
            rule = self._create_rule(source, target)

            result.append(rule)

        self._verify_rules(result)

        return tuple(result)

    def _generate_rules(
        self,
    ) -> t.Tuple[casebase.Rule, ...]:
        case_text = self.case_graph.text

        if self.config.rules_from_mc_only and (
            mc := self.case_graph.major_claim or self.case_graph.root_node
        ):
            case_text = mc.plain_text

        rules: dict[casebase.Rule, int] = {}
        case_kw = self.nlp.keywords([case_text], self.config.heuristic_pos_tags)
        query_kw = self.nlp.keywords(
            [self.query_graph.text], self.config.heuristic_pos_tags
        )

        for source_token, target_token in itertools.product(
            case_kw,
            query_kw,
        ):
            if (
                source_token.lemma != target_token.lemma
                and source_token.pos_tag == target_token.pos_tag
            ):
                with contextlib.suppress(AssertionError):
                    source = self._parse_concept(
                        adaptation_pb2.Concept(
                            lemma=source_token.lemma.strip(),
                            pos=casebase.spacy2pos(source_token.pos_tag),
                        ),
                        None,
                    )
                    target = self._parse_concept(
                        adaptation_pb2.Concept(
                            lemma=target_token.lemma.strip(),
                            pos=casebase.spacy2pos(target_token.pos_tag),
                        ),
                        source.atoms,
                    )

                    if (
                        paths := wordnet.all_shortest_paths(
                            source.synsets, target.synsets
                        )
                    ) and (distance := len(next(iter(paths)))):
                        rule = self._rule_from_shortest_paths(source, target, paths)
                        rules[rule] = distance

        if rules.values():
            min_distance = min(rules.values())

            # Return all rules that have the shortest distance in the knowledge graph.
            return tuple(
                rule
                for rule, distance in rules.items()
                if distance == min_distance and rule.source != rule.target
            )

        return tuple()

    def _verify_rules(self, rules: t.Collection[casebase.Rule]) -> None:
        if len(rules) != len({rule.source for rule in rules}):
            raise RuntimeError(
                "The number of rules specified does not match the number of unique"
                " lemmas in the source column. Please verify that you only specify one"
                " rule per lemma (e.g., 'runner/noun'), not one rule per form (e.g.,"
                " 'runner/noun' and 'runners/noun'). Different POS tags however should"
                " be represented with multiple rules (e.g. 'runner/noun' and"
                " 'running/verb')."
            )

    def _create_rule(
        self, source: casebase.Concept, target: casebase.Concept
    ) -> casebase.Rule[casebase.Concept]:
        if self.config.enforce_user_rule_paths:
            paths = wordnet.all_shortest_paths(source.synsets, target.synsets)

            if len(paths) == 0:
                # err = (
                #     f"The given rule '{str(source)}->{str(target)}'"
                #     " is invalid. No path to connect the concepts could be found in the"
                #     " knowledge graph. "
                # )

                # lemmas = itertools.chain.from_iterable(
                #     hyp.lemmas
                #     for synset in source.synsets
                #     for hyp, _ in synset.hypernym_distances().items()
                #     if not any(lemma.startswith(source.lemma) for lemma in hyp.lemmas)
                #     and hyp._synset.id not in config["wordnet"]["hypernym_filter"]
                # )

                # err += f"The following hypernyms are permitted: {sorted(lemmas)}"

                raise RuntimeError(
                    f"The given rule '{str(source)}->{str(target)}' is invalid. No path"
                    " to connect the concepts could be found in wordnet."
                )

            return self._rule_from_shortest_paths(source, target, paths)

        return casebase.Rule(source, target)

    def _rule_from_shortest_paths(
        self,
        source: casebase.Concept,
        target: casebase.Concept,
        paths: t.Collection[wordnet.Path],
    ) -> casebase.Rule:
        source_synsets = frozenset(path.start_node for path in paths)
        target_synsets = frozenset(path.end_node for path in paths)

        source = casebase.Concept.from_concept(source, synsets=source_synsets)
        target = casebase.Concept.from_concept(target, synsets=target_synsets)

        return casebase.Rule(source, target)

    def _parse_concept(
        self,
        concept: adaptation_pb2.Concept,
        atoms: t.Optional[t.FrozenSet[ag.AtomNode]],
    ) -> casebase.Concept:
        lemma, form2pos, pos2form = inflect_concept(
            concept.lemma, casebase.pos2spacy(concept.pos), lemmatize=True
        )

        if not atoms:
            _atoms = set()

            # Only accept rules that cover a complete word.
            # If for example 'landlord' is a rule, but the node only contains 'landlords',
            # an exception will be thrown.
            for form in form2pos:
                pattern = re.compile(f"\\b({form})\\b")

                for atom in self.case_graph.atom_nodes.values():
                    atom_txt = atom.plain_text.lower()

                    if pattern.search(atom_txt):
                        _atoms.add(atom)

            assert _atoms, (
                f"The concept '{concept.lemma}' with the forms '{form2pos}'"
                " could not be found in the argument graph. Please check the spelling."
            )

            atoms = frozenset(_atoms)

        synsets = self.wn.concept_synsets(
            form2pos.keys(),
            concept.pos,
            self.nlp,
            [atom.plain_text for atom in atoms],
            self.config.synset_similarity_threshold,
        )

        assert synsets, (
            f"The concept '{concept.lemma}' with the forms '{form2pos}'"
            " could not be found in wordnet."
        )

        return casebase.Concept(
            lemma,
            form2pos,
            pos2form,
            concept.pos,
            atoms,
            synsets,
        )
