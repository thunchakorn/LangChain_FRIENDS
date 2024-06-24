FROM python:3.12.4-slim-bookworm

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get install build-essential -y

WORKDIR /app

RUN addgroup --system langchain \
    && adduser --system --ingroup langchain langchain

COPY --chown=langchain:langchain requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=langchain:langchain . .

RUN chown -R langchain:langchain /app

USER langchain