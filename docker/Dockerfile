FROM python:3.8-slim-bullseye

WORKDIR /app

ENV PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ../src /app/src
COPY ../docs /app/docs
COPY ../artifacts /app/artifacts
COPY ../conf /app/conf
COPY ../requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 80

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]