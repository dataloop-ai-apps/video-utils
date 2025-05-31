FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.8.pytorch2

USER root
RUN export CUDA_HOME=/usr/local/cuda-11.8
RUN export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install system dependencies
RUN apt-get update && apt-get install -y cmake

USER 1000
WORKDIR /tmp
RUN git clone https://github.com/ifzhang/ByteTrack.git
WORKDIR /tmp/ByteTrack
RUN pip install -r requirements.txt
RUN python3 setup.py develop
RUN export PYTHONPATH=/tmp/ByteTrack:$PYTHONPATH

RUN git clone https://github.com/nwojke/deep_sort.git

RUN pip install dotenv

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

