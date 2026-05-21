FROM python:3.10-slim AS base

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

FROM base AS dev
CMD ["flask", "--app", "wsgi", "run", "--host=0.0.0.0", "--reload"]

FROM base AS prod
CMD ["/usr/local/bin/gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app", "-w", "1", "-t", "60"]
