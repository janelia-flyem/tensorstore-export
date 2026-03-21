# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required for TensorStore and Arrow
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Build BRAID C extension (dvid_decompress.so)
COPY braid/csrc/ braid/csrc/
RUN make -C braid/csrc

# Copy and install BRAID library
COPY braid/pyproject.toml braid/README.md braid/
COPY braid/src/ braid/src/
RUN pip install --no-cache-dir braid/

# Copy and install main app dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY main.py .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

ENV PYTHONPATH=/app

CMD ["python", "main.py"]
