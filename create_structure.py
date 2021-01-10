#!/usr/bin/env python3
import shutil
import typing as t
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(parent_folders: t.List[Path]):
    for parent_folder in parent_folders:
        cases = parent_folder.iterdir()

        for case in cases:
            if case.is_file() and case.suffix == ".json":
                case_graph = case.with_suffix(".pdf")

                nested_folder = parent_folder / case.stem
                nested_folder.mkdir()

                shutil.move(str(case), str(nested_folder / "graph.json"))

                if case_graph.exists():
                    shutil.move(str(case_graph), str(nested_folder / "graph.pdf"))


if __name__ == "__main__":
    app()
