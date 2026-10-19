# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY AIstg_v1.1.py .

# Create necessary directories
RUN mkdir -p models/finetuned OUTPUT logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NEWS_API_KEY=""

# Expose any ports if needed (currently not required for CLI app)
# EXPOSE 8000

# Default command (can be overridden)
# To run interactively: docker run -it --rm -e NEWS_API_KEY=your_key stock-analysis
# To run with custom command: docker run -it --rm stock-analysis python AIstg_v1.1.py
CMD ["python", "AIstg_v1.1.py"]
