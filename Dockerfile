# Base image
FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.10.pytorch2

# Set environment variables
USER root
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install system dependencies
RUN apt-get update && apt-get install -y cmake curl && rm -rf /var/lib/apt/lists/*

# Switch back to non-root user
USER 1000

WORKDIR /trackers
RUN git clone https://github.com/ifzhang/ByteTrack.git

# Set working directory
WORKDIR /tmp/
# Copy requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set PYTHONPATH for ByteTrack tracker
ENV PYTHONPATH="/trackers/ByteTrack:$PYTHONPATH"

# Install ByteTrack (needs root for setup.py develop)
USER root
WORKDIR /trackers/ByteTrack
RUN python3 setup.py develop
USER 1000

WORKDIR /


