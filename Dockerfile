# https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker

FROM python:3.8-slim
ENV POETRY_VERSION=1.1.4

WORKDIR /app

RUN apt update \
    && apt install -y --no-install-recommends graphviz \
    && apt install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}" \
    && poetry config virtualenvs.create false

COPY poetry.lock* pyproject.toml ./
RUN poetry install --no-interaction --no-ansi

RUN python -c "import wn; wn.download('ewn:2020')" \
    && python -m spacy download en_core_web_lg \
    && python -m spacy download en_core_web_sm
