# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required for TensorStore and Arrow
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    gcc \
    g++ \
    # Required for TensorStore
    cmake \
    # Required for some Python packages
    pkg-config \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Set Python path
ENV PYTHONPATH=/app

# Command to run the application
CMD ["python", "main.py"]