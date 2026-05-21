FROM python:3.10.16-slim

WORKDIR /app

RUN apt-get update && apt-get install -y wget gnupg && \
    echo "deb http://apt.postgresql.org/pub/repos/apt/ bookworm-pgdg main" > /etc/apt/sources.list.d/pgdg.list && \
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - && \
    apt-get update && apt-get install -y \
    libpq-dev gcc postgresql-client-16 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install Flask-Dance bs4 requests openai google-cloud-firestore tiktoken watchfiles
RUN pip install google-cloud-firestore google-cloud-storage langchain-community


# Copy the scripts first
COPY entrypoint.sh sync-db.sh ./
RUN sed -i 's/\r$//' /app/entrypoint.sh && \
    sed -i 's/\r$//' /app/sync-db.sh && \
    chmod +x /app/entrypoint.sh /app/sync-db.sh

# Then copy everything else
COPY . .

# Set environment variables
ENV POETRY_VIRTUALENVS_CREATE=false
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# RUN poetry install --no-root

# Expose the port the app runs on
EXPOSE 5000 8000

ENTRYPOINT ["bash", "/app/entrypoint.sh"]
CMD ["flask", "run", "--host=0.0.0.0"]