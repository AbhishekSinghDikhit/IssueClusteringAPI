# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pipenv / poetry optional
# COPY Pipfile Pipfile.lock ./  
# RUN pip install pipenv && pipenv install --deploy --ignore-pipfile  

# Copy requirements file first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole app
COPY . .

# Expose port (Render will map this automatically)
EXPOSE 8000

# Start the app with uvicorn (Render requires 0.0.0.0 binding)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]