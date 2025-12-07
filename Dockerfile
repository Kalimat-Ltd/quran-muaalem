FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# git is required for installing dependencies from git repositories
# libsndfile1 is required for audio processing (librosa/torchaudio)
# ffmpeg is required for loading mp3 and other audio formats
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Note: Installing torchaudio and librosa explicitly as they appear to be missing from requirements.txt
# but are imported in the code.
# python-multipart is required for FastAPI form data support.
RUN pip install -r requirements.txt && \
    pip install torch==2.8.0 torchvision torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Copy the rest of the application code
COPY . .

# Set PYTHONPATH to include the src directory
ENV PYTHONPATH=/app/src

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "src/fastapi_server.py"]
