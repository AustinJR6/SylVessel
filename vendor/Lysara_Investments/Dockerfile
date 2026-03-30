FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ALLOW_LIVE_TRADING_STARTUP=true \
    CONTROL_API_HOST=0.0.0.0 \
    CONTROL_API_PORT=18791

ARG CACHE_BUST=dev

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt

COPY . /app

EXPOSE 18791

CMD ["python", "main.py"]
