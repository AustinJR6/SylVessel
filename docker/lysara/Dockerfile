FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MOCK_LYSARA_HOST=0.0.0.0 \
    MOCK_LYSARA_PORT=18791

WORKDIR /app

COPY scripts/mock_lysara_ops.py /app/mock_lysara_ops.py

EXPOSE 18791

CMD ["python", "/app/mock_lysara_ops.py"]
