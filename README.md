# ReCAP Argument Graph Adaptation

## Installation

```sh
poetry install
cp config-example.toml config.toml
poetry run python -m spacy download en_core_web_lg
poetry run python -m spacy download en_core_web_sm
poetry run python -m nltk.downloader popular universal_tagset
```

## Running

- Start a spacy server in one terminal session:

  `./nlp.sh`

- Run the adaptation in another session:

  `poetry run python -m arguegen`

## Folder Structure

Within the folder `data`, there are two folders: `input` and `output`.

## Input Folder

For each case that should be adapted, create a folder with an arbitrary name in `input`.
Then, place the following files in that folder:

- `graph.json`: Argument graph that should be adapted.
- `query.txt`: Query for the imaginary retrieval.
- `rules.csv`: Adaptation rules for all concepts in `graph.json` in the form `source_concept/pos,target_concpept/pos`.

## Output Folder

You only need to create the folder `output`.
It will then be used by the application to store the adapted cases.
For each run, you will get a folder that is named based on the current date and time.
You are free to delete adapted cases that are not needed anymore.
