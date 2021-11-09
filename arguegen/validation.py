import itertools
import statistics
import typing as t
from pathlib import Path

import typer
from nltk.metrics import agreement, masi_distance

from arguegen.controller import load
from arguegen.model import casebase
from arguegen.model.config import Config

app = typer.Typer()


@app.command()
def rule_agreement(path: Path) -> None:
    config = Config.instance()
    config["loading"]["user_defined_rules"] = False
    cases = load.cases(path)

    total_rules = 0
    total_common_rules = 0
    total_contains_best_rule = 0
    total_contains_any_rule = 0

    for case in cases:
        # common_rules = set(user_case.rules).intersection(system_case.rules)
        # total_rules = set(user_case.rules).union(system_case.rules)

        common_rules = sum(rule in case.benchmark_rules for rule in case.rules)

        contains_best_rule = case.benchmark_rules[0] in case.rules
        contains_any_rule = common_rules > 0

        total_rules += len(case.benchmark_rules)
        total_common_rules += common_rules
        total_contains_best_rule += 1 if contains_best_rule else 0
        total_contains_any_rule += 1 if contains_any_rule else 0

        print(
            f"{case.relative_path} - generated: {[str(rule) for rule in case.rules]}, user: {[str(rule) for rule in case.benchmark_rules]}"
        )
        # print(
        #     f"{case.relative_path} - common rules: {common_rules}, system rules: {len(case.rules)}, expert rules: {len(case.benchmark_rules)}"
        # )

    print(f"Agreement over all rules: {total_common_rules}/{total_rules}")
    print(f"Agreement over best rule: {total_contains_best_rule}/{len(cases)}")
    print(f"Agreement over any rule: {total_contains_any_rule}/{len(cases)}")


@app.command()
def check_rules(path: Path) -> None:
    load.cases(path, ignore_errors=False)


def _triples(
    name: str,
    func: t.Callable[[casebase.Case], t.Iterable[str]],
    case1: casebase.Case,
    case2: casebase.Case,
) -> t.List[t.Tuple[str, str, t.FrozenSet[t.Any]]]:
    return [
        ("coder1", name, frozenset(func(case1))),
        ("coder2", name, frozenset(func(case2))),
    ]


@app.command()
def annotator_agreement(path1: Path, path2: Path) -> None:
    cases1 = load.cases(path1)
    cases2 = load.cases(path2)
    source_triples = set()
    target_triples = set()
    source_target_triples = set()
    first_source_triples = set()
    first_target_triples = set()
    first_source_target_triples = set()
    total_rules = 0
    rules_per_case = []
    rule_code = lambda x: f"{x.source.code},{x.target.code}"

    for c1, c2 in itertools.product(cases1, cases2):
        if c1.relative_path == c2.relative_path:
            name = str(c1.relative_path)

            total_rules += len(c1.benchmark_rules) + len(c2.benchmark_rules)
            rules_per_case.extend((len(c1.benchmark_rules), len(c2.benchmark_rules)))

            first_source_triples.update(
                _triples(
                    name,
                    lambda x: [x.benchmark_rules[0].source.code],
                    c1,
                    c2,
                )
            )
            first_target_triples.update(
                _triples(
                    name,
                    lambda x: [x.benchmark_rules[0].target.code],
                    c1,
                    c2,
                )
            )
            first_source_target_triples.update(
                _triples(
                    name,
                    lambda x: [rule_code(x.benchmark_rules[0])],
                    c1,
                    c2,
                )
            )

            source_triples.update(
                _triples(
                    name,
                    lambda x: (rule.source.code for rule in x.benchmark_rules),
                    c1,
                    c2,
                )
            )
            target_triples.update(
                _triples(
                    name,
                    lambda x: (rule.target.code for rule in x.benchmark_rules),
                    c1,
                    c2,
                )
            )
            source_target_triples.update(
                _triples(
                    name,
                    lambda x: (rule_code(rule) for rule in x.benchmark_rules),
                    c1,
                    c2,
                )
            )

    _echo_task("only sources", source_triples, True)
    _echo_task("only targets", target_triples, True)
    _echo_task("sources and targets", source_target_triples, True)
    _echo_task("first source per case", first_source_triples, False)
    _echo_task("first target per case", first_target_triples, False)
    _echo_task("first source and target per case", first_source_target_triples, False)

    typer.echo(f"total rules: {total_rules}")
    typer.echo(f"average rules per case: {statistics.mean(rules_per_case)}")


def _echo_task(
    message: str, triples: t.Collection[t.Tuple[str, str, str]], multi_assign: bool
) -> None:
    task = agreement.AnnotationTask(triples, masi_distance)
    annotations = len(triples)

    typer.echo(f"{message.capitalize()}, {multi_assign=}, {annotations=}")

    if not multi_assign:
        typer.echo(f"\tBennett's S: {task.S()}")
        typer.echo(f"\tScott's Pi: {task.pi()}")
        typer.echo(f"\tFleiss's Kappa: {task.multi_kappa()}")
        typer.echo(f"\tCohen's Kappa: {task.kappa()}")

    typer.echo(f"\tCohen's Weighted Kappa: {task.weighted_kappa()}")
    typer.echo(f"\tKrippendorff's Alpha: {task.alpha()}")

    typer.echo()


if __name__ == "__main__":
    app()
