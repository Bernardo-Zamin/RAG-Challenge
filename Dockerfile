# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 curl\
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code
COPY RAG-Challenge/ RAG-Challenge/

# Define the PYTHONPATH for the application
ENV PYTHONPATH=/app/RAG-Challenge

# Expose port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "RAG-Challenge.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
