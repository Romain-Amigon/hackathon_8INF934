FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive



RUN apt-get update && apt-get install -y \
    sqlite3 \
    libsqlite3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/agent/graph.py"]

RUN export LANGSMITH_TRACING="true"
