# Estágio de Build
FROM python:3.11-slim as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Estágio de execução final
FROM python:3.11-slim as runner

WORKDIR /app

ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app

COPY --from=builder /root/.local  /root/.local
COPY . /app

# Executa o treino caso o Modelo não exista
RUN python src/train.py

EXPOSE 8000

CMD ["unicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]