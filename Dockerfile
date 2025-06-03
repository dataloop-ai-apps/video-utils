# Base image
FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.8.pytorch2

# Set environment variables
USER root
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install system dependencies
RUN apt-get update && apt-get install -y cmake && rm -rf /var/lib/apt/lists/*

# Switch back to non-root user
USER 1000

WORKDIR /app/trackers
RUN git clone https://github.com/ZQPei/deep_sort_pytorch.git
RUN git clone https://github.com/ifzhang/ByteTrack.git
RUN git clone https://github.com/NirAharon/BoT-SORT.git


# Set working directory
WORKDIR /app
# Copy requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set PYTHONPATH for all trackers
ENV PYTHONPATH="/app/trackers/deep_sort_pytorch:/app/trackers/ByteTrack:/app/trackers/BoT-SORT:$PYTHONPATH"

# Install ByteTrack (needs root for setup.py develop)
USER root
WORKDIR /app/trackers/ByteTrack
RUN pip install -r requirements.txt && python3 setup.py develop
USER 1000

#TODO remove this
RUN pip install dotenv
RUN pip install dtlpy


