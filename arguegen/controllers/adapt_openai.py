import functools
import json
import logging
import traceback
import typing as t
from dataclasses import dataclass

import openai
from arg_services.cbr.v1beta import adaptation_pb2
from arg_services.cbr.v1beta.model_pb2 import AnnotatedGraph

from arguegen.config import ExtrasConfig
from arguegen.controllers import adapt, extract
from arguegen.controllers.loader import Loader
from arguegen.model import casebase
from arguegen.model.nlp import Nlp
from arguegen.model.wordnet import Wordnet

log = logging.getLogger(__name__)


class ChatMessage(t.TypedDict):
    role: t.Literal["user", "system", "assistant"]
    content: str


@dataclass()
class AdaptOpenAI:
    case_name: str
    req: adaptation_pb2.AdaptedCaseRequest
    query: AnnotatedGraph
    config: ExtrasConfig
    nlp: Nlp
    wn: Wordnet

    def __post_init__(self) -> None:
        pass

    @functools.cached_property
    def given_rules(self) -> t.Optional[str]:
        if len(self.req.rules) == 0:
            return None

        parsed_rules = ", ".join(
            f"adapt the {rule.source.pos} {rule.source.lemma} to the"
            f" {rule.target.pos} {rule.target.lemma}"
            for rule in self.req.rules
        )

        return f"""

            Please use the following rules as a starting point:
            {parsed_rules}
        """

    @functools.cached_property
    def original_texts(self) -> dict[str, str]:
        return {
            id: node.atom.text
            for id, node in filter(
                lambda x: x[1].WhichOneof("type") == "atom",
                self.req.case.graph.nodes.items(),
            )
        }

    def compute(
        self,
    ) -> adaptation_pb2.AdaptedCaseResponse:
        if self.config.type == "openai-edit":
            return self._edit()
        elif self.config.type in ("openai-chat-prose", "openai-chat-explainable"):
            return self._chat()
        elif self.config.type == "openai-chat-hybrid":
            return self._chat_hybrid()

        log.error(f"Invalid type selected: {self.config.type}")

        return adaptation_pb2.AdaptedCaseResponse()

    def _apply_adaptations(self, adapted_texts: t.Mapping[str, str]) -> AnnotatedGraph:
        adapted_case = AnnotatedGraph()
        adapted_case.CopyFrom(self.req.case)

        for node_id, text_adapted in adapted_texts.items():
            adapted_case.graph.nodes[node_id].atom.text = text_adapted
            adapted_case.text = adapted_case.text.replace(
                self.original_texts[node_id], text_adapted
            )

        return adapted_case

    def _edit(
        self,
    ) -> adaptation_pb2.AdaptedCaseResponse:
        adapted_texts: dict[str, str] = {}

        instruction = f"""
            A user entered the following query into an argument search engine:
            {self.query.text}

            You should now edit the text to make it more relevant to the presented query.
            Please only specialize or generalize the most important keywords in the text and do not rewrite the text.
            {self.given_rules if self.given_rules else ""}
        """

        for node_id, original_text in self.original_texts.items():
            res = openai.Edit.create(
                model=self.config.openai.edit_model,
                input=original_text,
                instruction=instruction,
            )

            adapted_texts[node_id] = res.choices[0].text  # type: ignore

        return adaptation_pb2.AdaptedCaseResponse(
            case=self._apply_adaptations(adapted_texts)
        )

    def _chat(
        self,
    ) -> adaptation_pb2.AdaptedCaseResponse:
        # https://github.com/openai/chatgpt-retrieval-plugin/blob/88d983585816b7f298edb0cabf7502c5ccff370d/services/extract_metadata.py#L11
        # https://github.com/openai/chatgpt-retrieval-plugin/blob/88d983585816b7f298edb0cabf7502c5ccff370d/services/pii_detection.py#L6

        adapted_texts: dict[str, str] = {}
        applied_rules: list[adaptation_pb2.Rule] = []

        system_message = f"""
            A user entered the following query into an argument search engine:
            {self.query.text}

            The search engine provided the user with the following result:
            {self.req.case.text}

            The user will now provide you with segments from that result that need to be adapted to make it more relevant to the presented query.
            You sould keep the changes as small as possible and only generalize the most important keywords in the text.
            {self.given_rules if self.given_rules else ""}
        """

        if self.config.type == "openai-chat-explainable":
            system_message += """
                Do not rewrite whole sentences or paragraphs, only replace small chunks!

                Respond with a JSON of the following structure:
                {
                    text: string, the adapted text
                    rules: list of objects, the replacements needed to transform the original text into the adapted text:  [
                        {
                            source: string, keyword that needs to be replaced,
                            target: string, generalized keyword that should be used instead,
                            pos: noun/verb/adjective/adverb, part of speech tag of the keyword,
                            importance: float between 0 and 1, how important the keyword is in the text
                        }
                    ]
                }
            """

        messages: list[ChatMessage] = [{"role": "system", "content": system_message}]

        for node_id, text_original in self.original_texts.items():
            messages.append({"role": "user", "content": text_original})

            res = openai.ChatCompletion.create(
                model=self.config.openai.chat_model, messages=messages
            )

            raw_completion = res.choices[0].message.content.strip()  # type: ignore
            messages.append(res.choices[0].message)  # type: ignore

            try:
                completion = json.loads(raw_completion)
                adapted_texts[node_id] = completion["text"]
                completion_rules = completion.get("rules", [])

                for rule in completion_rules:
                    try:
                        pos = adaptation_pb2.Pos.Value(f"pos_{rule['pos']}".upper())
                    except ValueError:
                        pos = adaptation_pb2.Pos.POS_UNSPECIFIED

                    applied_rules.append(
                        adaptation_pb2.Rule(
                            source=adaptation_pb2.Concept(
                                lemma=rule["source"],
                                pos=pos,
                                score=rule["importance"],
                            ),
                            target=adaptation_pb2.Concept(
                                lemma=rule["target"],
                                pos=pos,
                                score=rule["importance"],
                            ),
                        )
                    )

            except json.JSONDecodeError:
                adapted_texts[node_id] = raw_completion

        return adaptation_pb2.AdaptedCaseResponse(
            case=self._apply_adaptations(adapted_texts), applied_rules=applied_rules
        )

    def _chat_hybrid(self) -> adaptation_pb2.AdaptedCaseResponse:
        case = Loader(
            self.case_name,
            self.req.case,
            self.query,
            self.nlp,
            self.wn,
            self.config.loader,
        ).parse(self.req)
        extracted_concepts, discarded_concepts = extract.keywords(
            case, self.nlp, self.config.extraction, self.config.score, self.wn
        )

        adapted_rules = self._predict_rules(extracted_concepts)
        adapted_graph, applied_rules = adapt.argument_graph(
            case, adapted_rules, self.nlp, self.config.adaptation
        )

        return adaptation_pb2.AdaptedCaseResponse(
            case=adapted_graph.dump(),
            applied_rules=[rule.dump() for rule in applied_rules],
            extracted_concepts=[concept.dump() for concept in extracted_concepts],
            discarded_concepts=[concept.dump() for concept in discarded_concepts],
        )

    def _predict_rules(
        self, _extracted_concepts: t.AbstractSet[casebase.ScoredConcept]
    ) -> list[casebase.Rule[casebase.ScoredConcept]]:
        # We need a deterministic ordering
        extracted_concepts = list(_extracted_concepts)
        rules: list[casebase.Rule[casebase.ScoredConcept]] = []

        if len(extracted_concepts) == 0:
            return rules

        system_message = f"""
            A user entered the following query into an argument search engine:
            {self.query.text}

            The search engine provided the user with the following result:
            {self.req.case.text}

            The user will now provide you with a list of concepts and keywords from that result that need to be generalized to make it more relevant to the presented query.
            Pay attention to the provided part of speech tag (POS) and treat the provided score as an estimation of the keyword's importance to be generalized.
            {self.given_rules if self.given_rules else ""}

        """

        system_message += """
            Respond with a list of JSON object having the following structure:
            {
                source: string, the 'lemma' provided by the user
                target: string, the generalized 'lemma' that should be used instead
                index: int, the 'index' of the lemma in the user input
                importance: float between 0 and 1, how important the generalization is for the text
            }
        """

        user_message = json.dumps(
            [
                {
                    "lemma": concept.concept.lemma,
                    "pos_tag": (
                        adaptation_pb2.Pos.Name(concept.concept.pos)
                        .lower()
                        .removeprefix("pos_")
                        if concept.concept.pos
                        else None
                    ),
                    "index": i,
                    "importance": concept.score,
                }
                for i, concept in enumerate(extracted_concepts)
            ],
            ensure_ascii=False,
            indent=2,
        )

        messages: list[ChatMessage] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        res = openai.ChatCompletion.create(
            model=self.config.openai.chat_model, messages=messages
        )

        raw_completion = res.choices[0].message.content.strip()  # type: ignore

        try:
            completions = json.loads(raw_completion)

            for completion in completions:
                extracted_concept: casebase.ScoredConcept = extracted_concepts[
                    completion["index"]
                ]

                assert extracted_concept.concept.lemma == completion["source"]

                adapted_concept = casebase.ScoredConcept(
                    casebase.Concept.from_concept(
                        extracted_concept.concept, completion["target"]
                    ),
                    completion["importance"],
                )

                rules.append(casebase.Rule(extracted_concept, adapted_concept))

        except Exception:
            log.error(f"""
                traceback:
                {traceback.format_exc()}

                user:
                {user_message}

                assistant:
                {raw_completion}
            """)

        return rules
