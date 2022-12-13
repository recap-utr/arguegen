FROM python:3.9-slim

ENV POETRY_VERSION=1.3.1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt update \
    && apt install -y --no-install-recommends graphviz \
    && apt install -y build-essential \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}"
COPY poetry.lock* pyproject.toml ./
RUN poetry install --no-interaction --no-ansi

RUN poetry run python -m nltk.downloader popular universal_tagset \
    && poetry run python -m wn download oewn:2021 \
    && poetry run python -m spacy download en_core_web_lg \
    && poetry run python -m spacy download en_core_web_sm

CMD [ "poetry", "run", "python",  "-m", "arguegen" ]
