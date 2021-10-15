FROM nvidia/cuda:10.2-runtime

RUN apt update
RUN apt install -y python3-pip python3.8-dev python3.8-venv python3-cffi python3-venv htop tmux python3-wheel python3-protobuf python3-six
RUN python3.8 -m pip install --no-cache-dir --upgrade wheel cffi
RUN python3.8 -m pip install --no-cache-dir torch torchvision

# Force the right torch_scatter
RUN python3.8 -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html

COPY ./requirements.txt requirements.txt
RUN python3.8 -m pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt



RUN mkdir -p /src/buglab
RUN mkdir -p /data/targetDir

WORKDIR /src/
ENV PYTHONPATH=/src/

RUN python3.8 -m pip install --upgrade --no-cache-dir py-spy


COPY buglab /src/buglab
WORKDIR /home/buglab
