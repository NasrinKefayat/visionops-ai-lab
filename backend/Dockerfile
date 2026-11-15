FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/pyproject.toml ./pyproject.toml
COPY backend/src ./src
RUN pip install --no-cache-dir . && useradd --create-home app && chown -R app:app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "visionops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
