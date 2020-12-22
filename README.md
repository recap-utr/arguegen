# ReCAP Argument Graph Adaptation

## Limitations

- On Windows-based systems, multiprocessing cannot be used currently. Set the config option `processes` to 1!

## Installation

```sh
poetry install
cp config-example.toml config.toml
poetry run python -m recap_argument_graph_adaptation
```

## Folder Structure

Within the folder `data`, there are two folders: `input` and `output`.

## Input Folder

For each case that should be adapted, create a folder with an arbitrary name in `input`.
Then, place the following files in that folder:

-   `benchmark.csv`
-   `benchmark.json`
-   `case.csv`
-   `case.json`
-   `query.txt`

## Output Folder

You only need to create the folder `output`.
It will then be used by the application to store the adapted cases.
For each run, you will get a folder that is named based on the current date and time.
You are free to delete adapted cases that are not needed anymore.
