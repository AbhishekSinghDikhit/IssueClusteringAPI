# Use a more robust Python base image to avoid dependency issues
FROM python:3.11

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Update package lists
RUN apt-get update && apt-get upgrade -y

# Install system dependencies
RUN apt-get install -y --no-install-recommends \
    libblas-dev \
    libatlas-base-dev \
    gfortran \
    libssl-dev \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for security
RUN useradd -m appuser
USER appuser

# Copy the entire application
COPY . .

# Expose port (Render maps this automatically)
EXPOSE 8000

# Add healthcheck for FastAPI
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the app with uvicorn (Render requires 0.0.0.0 binding)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]