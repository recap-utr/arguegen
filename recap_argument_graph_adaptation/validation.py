import itertools
import typing as t
from pathlib import Path

import typer
from nltk.metrics import agreement

from recap_argument_graph_adaptation.controller import load

app = typer.Typer()


@app.command()
def check_rules(path: Path) -> None:
    load.cases(path, ignore_errors=True)


def _triples(name: str, value1: str, value2: str) -> t.List[t.Tuple[str, str, str]]:
    return [("coder1", name, value1), ("coder2", name, value2)]


@app.command()
def annotator_agreement(path1: Path, path2: Path) -> None:
    cases1 = load.cases(path1)
    cases2 = load.cases(path2)
    source_triples = set()
    target_triples = set()
    source_target_triples = set()
    common_target_triples = set()
    first_source_triples = set()
    rule_code = lambda x: f"{x.source.code},{x.target.code}"

    for c1, c2 in itertools.product(cases1, cases2):
        if c1.relative_path == c2.relative_path:
            name = str(c1.relative_path)

            first_source_triples.update(
                _triples(
                    name,
                    c1.benchmark_rules[0].source.code,
                    c2.benchmark_rules[0].source.code,
                )
            )

            for r1, r2 in itertools.product(c1.benchmark_rules, c2.benchmark_rules):
                source_triples.update(_triples(name, r1.source.code, r2.source.code))
                source_target_triples.update(
                    _triples(name, rule_code(r1), rule_code(r2))
                )

                if r1.source == r2.source:
                    common_target_triples.update(
                        _triples(
                            f"{name}_{r1.source.code}", r1.target.code, r2.target.code
                        )
                    )

    _echo_task("only sources", source_triples, True)
    _echo_task("sources and targets", source_target_triples, True)
    _echo_task("targets from common sources", common_target_triples, False)
    _echo_task("first source per case", first_source_triples, False)


def _echo_task(
    message: str, triples: t.Collection[t.Tuple[str, str, str]], multi_assign: bool
) -> None:
    task = agreement.AnnotationTask(triples)
    annotations = len(triples)

    typer.echo(f"{message.capitalize()}, {multi_assign=}, {annotations=}")

    typer.echo(f"\tKrippendorff's Alpha: {task.alpha()}")

    if not multi_assign:
        typer.echo(f"\tCohen's Kappa: {task.kappa()}")
        typer.echo(f"\tFleiss's Kappa: {task.multi_kappa()}")
        typer.echo(f"\tScott's Pi: {task.pi()}")

    typer.echo()


if __name__ == "__main__":
    app()
