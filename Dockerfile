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

WORKDIR /trackers
RUN git clone https://github.com/ZQPei/deep_sort_pytorch.git
RUN git clone https://github.com/ifzhang/ByteTrack.git
# Download the checkpoint file for deep_sort_pytorch
WORKDIR /trackers/deep_sort_pytorch_ckpt
RUN curl -L "https://drive.usercontent.google.com/download?id=1_qwTWdzT9dWNudpusgKavj_4elGgbkUN&export=download&authuser=0&confirm=t&uuid=66474786-6d1e-4645-b02f-f3efca1441e0&at=ALoNOgllhCQYv3CXbS8RWuVC4BaF:1749034406784" -o ckpt.t7

#RUN git clone https://github.com/NirAharon/BoT-SORT.git


# Set working directory
WORKDIR /tmp/
# Copy requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set PYTHONPATH for all trackers
ENV PYTHONPATH="/trackers/deep_sort_pytorch:/trackers/ByteTrack:$PYTHONPATH"

# Install ByteTrack (needs root for setup.py develop)
USER root
WORKDIR /trackers/ByteTrack
RUN pip install -r requirements.txt && python3 setup.py develop
USER 1000

WORKDIR /

#TODO remove this
RUN pip install dotenv
RUN pip install dtlpy


