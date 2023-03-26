import json
import logging
import traceback
import typing as t

import arg_services
import grpc
import openai
import rich_click as click
import typed_settings as ts
from arg_services.cbr.v1beta import adaptation_pb2, adaptation_pb2_grpc

from arguegen.config import AdaptationMethod, ExtrasConfig
from arguegen.controllers import adapt, extract, loader
from arguegen.model import wordnet
from arguegen.model.nlp import Nlp

openai.api_key_path = "./openai_api_key.txt"

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class ChatMessage(t.TypedDict):
    role: t.Literal["user", "system", "assistant"]
    content: str


@ts.settings(frozen=True)
class ServerConfig:
    address: str = "localhost:50300"
    nlp_address: str = "localhost:50100"
    threads: int = 1


class AdaptationService(adaptation_pb2_grpc.AdaptationServiceServicer):
    def __init__(self, server_config: ServerConfig):
        self.server_config = server_config

    def Adapt(
        self, req: adaptation_pb2.AdaptRequest, ctx: grpc.ServicerContext
    ) -> adaptation_pb2.AdaptResponse:
        log.debug(f"[{id(self)}] Processing request...")

        if req.extras["type"] == "openai":
            return self._adapt_openai(req)

        elif req.extras["type"] == "wordnet":
            return self._adapt_wordnet(req)

        return adaptation_pb2.AdaptResponse()

    # Prompts: https://github.com/openai/openai-cookbook/blob/main/techniques_to_improve_reliability.md
    def _adapt_openai(
        self, req: adaptation_pb2.AdaptRequest
    ) -> adaptation_pb2.AdaptResponse:
        adapted_cases = {}
        config = ExtrasConfig.from_extras(req.extras)

        for case_name, case_req in req.cases.items():
            case_adapted = case_req.case
            applied_rules = set()
            given_rules = ", ".join(
                f"adapt the {rule.source.pos} {rule.source.lemma} to the"
                f" {rule.target.pos} {rule.target.lemma}"
                for rule in case_req.rules
            )

            for node in case_adapted.graph.nodes.values():
                if node.WhichOneof("type") == "atom":
                    text_original = node.atom.text
                    text_adapted: t.Optional[str] = None

                    if config.openai.endpoint == "edit":
                        instruction = f"""
                            Imagine a user entered the following query into a search engine specialized in finding arguments:
                            {req.query.text}

                            You should now edit the text to make it more relevant to the presented query.
                            Please only specialize or generalize the most important keywords in the text and do not rewrite the text.
                        """

                        if given_rules:
                            instruction += (
                                "\n\nPlease use the following rules as a starting"
                                f" point:\n{given_rules}."
                            )

                        res = openai.Edit.create(
                            model="text-davinci-edit-001",
                            input=case_req.case.text,
                            instruction=instruction,
                        )

                        text_adapted = res.choices[0].text  # type: ignore

                    elif config.openai.endpoint == "chat":
                        # https://github.com/openai/chatgpt-retrieval-plugin/blob/88d983585816b7f298edb0cabf7502c5ccff370d/services/extract_metadata.py#L11
                        # https://github.com/openai/chatgpt-retrieval-plugin/blob/88d983585816b7f298edb0cabf7502c5ccff370d/services/pii_detection.py#L6
                        system_message: ChatMessage = {
                            "role": "system",
                            "content": f"""
                                A user entered the following query into a search engine specialized in finding arguments:
                                {req.query.text}

                                The search engine provided the user with the following result:
                                {case_req.case.text}

                                The user will now provide you with segments from that result that need to be adapted to make it more relevant to the presented query.
                                You sould keep the changes as small as possible and only specialize or generalize the most important keywords in the text.

                                Respond with a JSON containing key value pairs.
                                Use the following structure:
                                - text: string, the adapted text
                                - rules: list of objects in the form {{"source": string, "target": string, "pos": string (either 'NOUN', 'VERB', 'ADJECTIVE', or 'ADVERB'), "importance": float (between 0 and 1)}}, the text replacements needed to transform the original text into the adapted text together with their part of speech (pos) tags and their perceived importance for the adaptation
                            """,
                        }

                        user_message: ChatMessage = {
                            "role": "user",
                            "content": f"Here is the text segment:\n{text_original}",
                        }

                        if given_rules:
                            user_message["content"] += (
                                "\n\nPlease use the following rules as a starting"
                                f" point:\n{given_rules}."
                            )

                        res = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[system_message, user_message],
                        )

                        raw_completion = res.choices[0].message.content.strip()  # type: ignore

                        try:
                            completion = json.loads(raw_completion)
                            print(completion)
                            text_adapted = completion.get("text")

                            if completion_rules := completion.get("rules"):
                                for rule in completion_rules:
                                    applied_rules.add(
                                        adaptation_pb2.Rule(
                                            source=adaptation_pb2.Concept(
                                                lemma=rule["source"],
                                                pos=rule["pos"],
                                                score=rule["importance"],
                                            ),
                                            target=adaptation_pb2.Concept(
                                                lemma=rule["target"],
                                                pos=rule["pos"],
                                                score=rule["importance"],
                                            ),
                                        )
                                    )

                        except Exception:
                            print(traceback.format_exc())

                    if text_adapted is not None:
                        node.atom.text = text_adapted
                        case_adapted.text = case_adapted.text.replace(
                            text_original, text_adapted
                        )

            adapted_cases[case_name] = adaptation_pb2.AdaptedCaseResponse(
                case=case_adapted, applied_rules=applied_rules
            )

        return adaptation_pb2.AdaptResponse(cases=adapted_cases)

    def _adapt_wordnet(
        self, req: adaptation_pb2.AdaptRequest
    ) -> adaptation_pb2.AdaptResponse:
        adapted_cases = {}
        nlp = Nlp(self.server_config.nlp_address, req.nlp_config)
        wn = wordnet.Wordnet(nlp)

        config = ExtrasConfig.from_extras(req.extras)

        for case_name, case_req in req.cases.items():
            case = loader.Loader(
                case_name, case_req.case, req.query, nlp, wn, config.loader
            ).parse(case_req)
            extracted_concepts, discarded_concepts = extract.keywords(
                case, nlp, config.extraction, config.score, wn
            )

            adapted_rules = []

            if config.adaptation.method == AdaptationMethod.DIRECT:
                adapted_rules, rule_candidates = adapt.concepts(
                    extracted_concepts, case, nlp, config.adaptation, config.score
                )

            elif config.adaptation.method == AdaptationMethod.BFS:
                extracted_paths = extract.paths(
                    extracted_concepts, case.rules, config.adaptation.bfs_method
                )
                adapted_rules, adapted_paths, rule_candidates = adapt.paths(
                    extracted_paths, case, nlp, config.adaptation, config.score
                )

            adapted_graph, applied_rules = adapt.argument_graph(
                case, adapted_rules, nlp, config.adaptation
            )
            discarded_rules = set(adapted_rules).difference(applied_rules)

            adapted_cases[case_name] = adaptation_pb2.AdaptedCaseResponse(
                case=adapted_graph.dump(),
                applied_rules=[rule.dump() for rule in applied_rules],
                discarded_rules=[rule.dump() for rule in discarded_rules],
                extracted_concepts=[concept.dump() for concept in extracted_concepts],
                discarded_concepts=[concept.dump() for concept in discarded_concepts],
                # TODO: Add rule candidates
            )

        return adaptation_pb2.AdaptResponse(cases=adapted_cases)


class ServiceAdder:
    def __init__(self, config: ServerConfig):
        self.config = config

    def __call__(self, server: grpc.Server):
        adaptation_pb2_grpc.add_AdaptationServiceServicer_to_server(
            AdaptationService(self.config), server
        )


@click.command("arguegen")
@ts.click_options(ServerConfig, "server")
def app(config: ServerConfig):
    arg_services.serve(
        config.address,
        ServiceAdder(config),
        [arg_services.full_service_name(adaptation_pb2, "AdaptationService")],
        threads=config.threads,
    )


if __name__ == "__main__":
    app()
