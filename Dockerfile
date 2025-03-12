# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*  # Cleanup to reduce image size

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip and install dependencies in a single step to optimize caching
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt  

# Expose a port (optional, useful for web apps)
EXPOSE 5000

# Run your Python script when the container launches
CMD ["python", "DataFeatureExtraction.py"]
