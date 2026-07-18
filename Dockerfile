FROM python:3.10.19-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1

WORKDIR /app
COPY requirements/runtime.lock.txt requirements/runtime.lock.txt
COPY pyproject.toml README.md ./
COPY deepvac ./deepvac
COPY gru ./gru
COPY lstm ./lstm
COPY optimization ./optimization
COPY tcp ./tcp

# Runtime-only image: chamber protocol, PID/BO, and inference with an
# already-trained checkpoint. For training use requirements/training.lock.txt
# (see requirements/README.md).
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements/runtime.lock.txt \
    && python -m pip install --no-cache-dir --no-deps -e ".[runtime]" \
    && python -m pip check

CMD ["python", "-m", "optimization.list_runs", "--help"]
