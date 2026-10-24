# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY adk_agents ./adk_agents
COPY data ./data

# Run as non-root; Cloud Run injects $PORT.
RUN useradd --uid 10001 --create-home svc && chown -R svc /srv
USER svc
ENV PORT=8080 INDEX_PATH=/tmp/index/index.json
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn app.server:app --host 0.0.0.0 --port ${PORT} --workers 1"]
