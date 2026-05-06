FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.0

WORKDIR /app

COPY ml/requirements.txt /app/ml/requirements.txt
RUN pip install -r /app/ml/requirements.txt

COPY ml/ /app/ml/
COPY preambles/ /app/preambles/

RUN mkdir -p /app/data

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

EXPOSE 8000

CMD ["uvicorn", "ml.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
