import multiprocessing
import subprocess
import socket
import time

import requests

import uvicorn

from recap_argument_graph_adaptation.model.config import Config

config = Config.instance()

request_exceptions = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.HTTPError,
)


def _get_open_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


# def run() -> None:
#     host = config["spacy"]["host"]
#     port = config["spacy"]["port"]
#     workers = config["spacy"]["workers"]

#     server = multiprocessing.Process(
#         target=uvicorn.run,
#         args=("recap_argument_graph_adaptation.spacy_server_rest:app",),
#         kwargs={
#             "host": host,
#             "port": port,
#             "log_level": "info",
#             "workers": workers,
#         },
#         daemon=False,
#     )
#     server.start()
#     server_ready = False

#     while not server_ready:
#         try:
#             response = requests.get(f"http://{host}:{port}")
#             response.raise_for_status()
#         except request_exceptions:
#             time.sleep(1)
#         else:
#             server_ready = True
#             print("Ready.")


def run() -> None:
    uvicorn.run(
        "recap_argument_graph_adaptation.spacy_server:app",
        host=config["spacy"]["host"],
        port=config["spacy"]["port"],
        log_level="warning",
        workers=config["spacy"]["workers"],
    )


if __name__ == "__main__":
    run()
