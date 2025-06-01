# Base image
FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.8.pytorch2

# Set environment variables
USER root
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install system dependencies
RUN apt-get update && apt-get install -y cmake && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Switch back to non-root user
USER 1000

# Copy files
COPY requirements.txt .
COPY trackers /tmp/trackers

# Install Python dependencies
RUN pip install -r requirements.txt \
    && pip install dotenv \
    && pip install dtlpy \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set PYTHONPATH for all trackers
ENV PYTHONPATH="/tmp/trackers:/tmp/trackers/deep_sort_pytorch:/tmp/trackers/ByteTrack:$PYTHONPATH"

# Install BoT-SORT requirements
WORKDIR /tmp/trackers/BoT_SORT
RUN pip install -r requirements.txt

# Install ByteTrack (needs root for setup.py develop)
USER root
WORKDIR /tmp/trackers/ByteTrack
RUN pip install -r requirements.txt && python3 setup.py develop
USER 1000

# Final working directory
WORKDIR /app
