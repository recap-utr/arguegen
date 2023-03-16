import logging
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

        for case_name, case_req in req.cases.items():
            case_adapted = case_req.case
            rules = ", ".join(
                f"adapt the {rule.source.pos} {rule.source.lemma} to the"
                f" {rule.target.pos} {rule.target.lemma}"
                for rule in case_req.rules
            )

            for node in case_adapted.graph.nodes.values():
                if node.WhichOneof("type") == "atom":
                    text_original = node.atom.text
                    text_adapted: t.Optional[str] = None

                    if req.extras["endpoint"] == "edit":
                        instruction = f"""
                            Imagine a user entered the following query into a search engine specialized in finding arguments:
                            {req.query.text}

                            You should now edit the text to make it more relevant to the presented query.
                            Please only specialize or generalize the most important keywords in the text and do not rewrite the text.
                        """

                        if rules:
                            instruction += (
                                "\n\nYou are giving the following rules as a starting"
                                f" point:\n{rules}."
                            )

                        res = openai.Edit.create(
                            model="text-davinci-edit-001",
                            input=case_req.case.text,
                            instruction=instruction,
                        )

                        text_adapted = res.choices[0].text  # type: ignore

                    elif req.extras["endpoint"] == "chat":
                        base_messages: tuple[ChatMessage, ...] = (
                            {
                                "role": "system",
                                "content": (
                                    "You are ChatGPT, a large language model trained by"
                                    " OpenAI. Answer as concisely as possible. Think"
                                    " step by step and keep the changes to the text"
                                    " minimal."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    "I would like to find arguments that are relevant"
                                    f" to the following query: {req.query.text}"
                                ),
                            },
                            {"role": "assistant", "content": case_req.case.text},
                        )

                        next_message: ChatMessage = {
                            "role": "user",
                            "content": f"""
                                Your result is already quite good, but needs some adjustments.
                                Please edit the following segment of the result to make it more relevant to the presented query.
                                Please only specialize or generalize the most important keywords in the text and do not rewrite the text.
                                Here is the segment:
                                {text_original}
                            """,
                        }

                        if rules:
                            next_message["content"] += (
                                "\n\nYou are giving the following rules as a starting"
                                f" point:\n{rules}."
                            )

                        res = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[*base_messages, next_message],
                        )

                        text_adapted = res.choices[0].message.content  # type: ignore

                        # We will now perform another request to retrieve the changed keywords
                        # messages.append(res.choices[0].message)  # type: ignore
                        # messages.append(
                        #     {
                        #         "role": "user",
                        #         "content": (
                        #             "Please list all changes as a list in the form of [source,"
                        #             " target]."
                        #         ),
                        #     }
                        # )

                        # res = openai.ChatCompletion.create(
                        #     model="gpt-3.5-turbo", messages=messages
                        # )
                        # print(res.choices[0].message.content)  # type: ignore

                    if text_adapted is not None:
                        node.atom.text = text_adapted
                        case_adapted.text = case_adapted.text.replace(
                            text_original, text_adapted
                        )

            adapted_cases[case_name] = adaptation_pb2.AdaptedCaseResponse(
                case=case_adapted,
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
