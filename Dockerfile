# Use Python 3.11 slim image (lightweight and compatible)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p models uploads data/retrain app/templates app/static

# Expose port (Render expects port 10000, but we'll use dynamic PORT)
EXPOSE 10000

# Run the application with gunicorn (production server)
# Uses PORT environment variable from Render, defaults to 10000
# Reduced workers to 2 for memory constraints on free tier
CMD gunicorn -b 0.0.0.0:${PORT:-10000} --timeout 120 --workers 2 app.app:app