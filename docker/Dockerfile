FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps for GeoPandas / rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev gdal-bin libgeos-dev libproj-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard]

COPY . .
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.cache/torch

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
