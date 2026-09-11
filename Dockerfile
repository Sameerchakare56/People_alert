# Use lightweight Python 3.10 image
FROM python:3.10-slim

# Install system dependencies required for OpenCV / FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and assets
COPY . .

# Create uploads directory and set permissions
RUN mkdir -p uploads && chmod 777 uploads

# Expose ports for Hugging Face (7860) and standard cloud hosts (8000)
EXPOSE 7860
EXPOSE 8000

# Command to run the application
CMD ["python", "det-alert.py"]
