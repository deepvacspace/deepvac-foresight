FROM python:3.10.19-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY gru ./gru
COPY lstm ./lstm
COPY optimization ./optimization
COPY tcp ./tcp
COPY utils ./utils

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir . \
    && python -m pip check

CMD ["python", "-m", "optimization.list_runs", "--help"]
