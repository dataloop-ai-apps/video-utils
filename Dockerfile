#docker run -it --user root -v "D:\git_repos\video-utils:/app" dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.8.pytorch2

FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.8.pytorch2

USER root
RUN export CUDA_HOME=/usr/local/cuda-11.8
RUN export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install system dependencies
RUN apt-get update && apt-get install -y cmake

USER 1000
COPY requirements.txt .
COPY trackers /tmp/
RUN pip install -r requirements.txt

# TODO: remove this
RUN pip install dotenv

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install dtlpy

RUN export PYTHONPATH=/tmp/trackers/deep_sort_pytorch:$PYTHONPATH


WORKDIR /tmp/trackers/BoT_SORT/
RUN pip install -r requirements.txt
RUN export PYTHONPATH=/tmp/trackers/BoT_SORT:$PYTHONPATH

WORKDIR /tmp/trackers/ByteTrack
RUN pip install -r requirements.txt
RUN python3 setup.py develop
RUN export PYTHONPATH=/tmp/trackers/ByteTrack:$PYTHONPATH


