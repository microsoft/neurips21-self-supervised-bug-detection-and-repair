FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn7-ubuntu18.04

RUN apt update
RUN apt install -y python3-pip python3.8-dev python3.8-venv python3-cffi python3-venv htop tmux python3-wheel python3-protobuf python3-six python3-cryptography
RUN apt-get install --yes cmake pkg-config build-essential
RUN python3.8 -m pip install --no-cache-dir torch==1.10.0
RUN python3.8 -m pip install --no-cache-dir wheel cffi cython

RUN python3.8 -m  pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.0+cu102.html

RUN python3.8 -m pip install --upgrade pip

RUN python3.8 -m pip install --no-cache-dir annoy chardet docopt msgpack opentelemetry-api==0.17b0 opentelemetry-exporter-jaeger==0.17b0 opentelemetry-exporter-prometheus==0.17b0 opentelemetry-sdk==0.17b0 prometheus-client ptgnn dpu-utils pyyaml pyzmq tqdm typing_extensions
RUN python3.8 -m pip install azureml-sdk ogb numba
