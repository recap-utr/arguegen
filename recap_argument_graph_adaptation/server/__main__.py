import multiprocessing

import typer
import uvicorn
from recap_argument_graph_adaptation.model.config import Config

config = Config.instance()
app = typer.Typer()

# def _get_open_port() -> int:
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.bind(("", 0))
#     s.listen(1)
#     port = s.getsockname()[1]
#     s.close()
#     return port

default_args = {
    "log_level": "info",
    "access_log": False,
}


@app.command(help="Start a spacy server.")
def spacy():
    uvicorn.run(
        "recap_argument_graph_adaptation.server.spacy_server:app",
        **{
            "host": config["resources"]["spacy"]["host"],
            "port": config["resources"]["spacy"]["port"],
            "workers": config["resources"]["spacy"]["workers"],
        },
        **default_args
    )


@app.command(help="Start a WordNet server.")
def wordnet():
    uvicorn.run(
        "recap_argument_graph_adaptation.server.wordnet_server:app",
        **{
            "host": config["resources"]["wordnet"]["host"],
            "port": config["resources"]["wordnet"]["port"],
            "workers": config["resources"]["wordnet"]["workers"],
        },
        **default_args
    )


if __name__ == "__main__":
    app()
