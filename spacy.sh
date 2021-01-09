#!/usr/bin/env sh
CUDA_VISIBLE_DEVICES="" poetry run python -m recap_argument_graph_adaptation.uvicorn spacy
