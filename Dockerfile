FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime 

USER $UID
ENV USER=docker

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y xauth zsh git curl glmark2 lsb-release gnupg2 wget vim gcc g++ tmux swig libgl1 libglib2.0-0 libsm6 libxrender1 libxext6

COPY . sheeprl/

RUN cd sheeprl && \
    pip install -e . && \
    pip install .[atari,box2d,mujoco,dev,test] && \
    pip install protobuf==4.25.3 # fix tensorboard issue

WORKDIR /workspace/sheeprl