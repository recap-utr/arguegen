import logging

import arg_services
import grpc
import rich_click as click
import typed_settings as ts
from arg_services.cbr.v1beta import adaptation_pb2, adaptation_pb2_grpc

from arguegen.config import AdaptationMethod, ExtrasConfig
from arguegen.controllers import adapt, extract, loader
from arguegen.model import wordnet
from arguegen.model.nlp import Nlp

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


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
