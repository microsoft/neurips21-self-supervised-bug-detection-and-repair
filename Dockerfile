FROM mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04

RUN apt update
RUN apt install -y python3-pip python3.8-dev python3.8-venv python3-cffi python3-venv htop tmux python3-wheel python3-protobuf python3-six python3-cryptography
RUN python3.8 -m pip install --no-cache-dir wheel cffi cython
RUN python3.8 -m pip install --no-cache-dir torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# ReInstall torch scatter
RUN python3.8 -m pip install --no-cache-dir --upgrade torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html

# Test installation
# RUN python3.8 -c "import torch_scatter"

RUN python3.8 -m pip install --no-cache-dir annoy chardet docopt msgpack opentelemetry-api==0.17b0 opentelemetry-exporter-jaeger==0.17b0 opentelemetry-exporter-prometheus==0.17b0 opentelemetry-sdk==0.17b0 prometheus-client ptgnn dpu-utils pyyaml pyzmq tqdm typing_extensions
RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install azureml-sdk
