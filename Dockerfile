# Dockerfile (CPU) 
FROM python:3.11-slim

# system deps required by HF/tokenizers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ca-certificates libsndfile1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV APP_HOME=/app
WORKDIR $APP_HOME

# copy requirements first for layer caching
COPY requirements.txt ./

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY src/ ./src/



EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
