ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim

ARG EXTRAS=""
ARG POETRY_VERSION=1.4.2
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt update \
    && apt install --no-install-recommends -y build-essential graphviz \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}" \
    && poetry config virtualenvs.in-project true

COPY poetry.lock* pyproject.toml ./
RUN poetry install --no-interaction --no-ansi --no-root

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN python -m nltk.downloader popular universal_tagset

COPY . ./

ENTRYPOINT [ "python", "-m", "arguegen" ]
CMD [ ]
