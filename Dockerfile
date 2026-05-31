FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app/backend

# Install system dependencies needed by faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "pymilvus[milvus_lite]>=2.4.0"

COPY backend/ .

EXPOSE 8000

CMD ["python", "main.py"]
