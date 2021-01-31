import itertools
from pathlib import Path

import typer
from nltk.metrics import agreement

from recap_argument_graph_adaptation.controller import load

app = typer.Typer()


@app.command()
def check_rules(path: Path) -> None:
    load.cases(path)
    typer.echo("No issues found.")


@app.command()
def annotator_agreement(path1: Path, path2: Path) -> None:
    cases1 = load.cases(path1)
    cases2 = load.cases(path2)
    source_triples = []
    target_triples = []
    source_target_triples = []
    common_target_triples = []
    rule_code = lambda x: f"{x.source.code},{x.target.code}"

    for c1, c2 in itertools.product(cases1, cases2):
        if c1.relative_path == c2.relative_path:
            name = str(c1.relative_path)

            for r1, r2 in itertools.product(c1.benchmark_rules, c2.benchmark_rules):
                source_triples.append(("coder1", name, r1.source.code))
                source_triples.append(("coder2", name, r2.source.code))

                target_triples.append(("coder1", name, r1.target.code))
                target_triples.append(("coder2", name, r2.target.code))

                source_target_triples.append(("coder1", name, rule_code(r1)))
                source_target_triples.append(("coder2", name, rule_code(r2)))

                if r1.source == r2.source:
                    common_target_triples.append(("coder1", name, r1.target.code))
                    common_target_triples.append(("coder2", name, r2.target.code))

    source_task = agreement.AnnotationTask(source_triples)
    target_task = agreement.AnnotationTask(target_triples)
    source_target_task = agreement.AnnotationTask(source_target_triples)
    common_target_task = agreement.AnnotationTask(common_target_triples)

    typer.echo(f"Considering only sources ({len(source_triples)} triples):")
    _echo_task(source_task)

    typer.echo()

    typer.echo(f"Considering only targets ({len(target_triples)} triples):")
    _echo_task(target_task)

    typer.echo()

    typer.echo(
        f"Considering both sources and targets ({len(source_target_triples)} triples):"
    )
    _echo_task(source_target_task)

    typer.echo()

    typer.echo(
        f"Considering only targets from common sources ({len(common_target_triples)} triples):"
    )
    _echo_task(common_target_task)


def _echo_task(task: agreement.AnnotationTask) -> None:
    typer.echo(f"\tKrippendorff's Alpha: {task.alpha()}")
    typer.echo(f"\tCohen's Kappa: {task.kappa()}")
    typer.echo(f"\tFleiss's Kappa: {task.multi_kappa()}")
    typer.echo(f"\tScott's Pi: {task.pi()}")


if __name__ == "__main__":
    app()
