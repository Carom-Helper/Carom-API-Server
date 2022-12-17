# cuda 11.6
FROM nvcr.io/nvidia/tensorrt:22.05-py3
# furiosa-sdk 0.8.0
# FROM 31f3b6f28041

SHELL ["/bin/bash", "-c"]

# Setup user account
ARG fname

# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive

# install
RUN apt-get update && apt-get install -y sudo \
    && apt-get install -y libgl1-mesa-glx git locales \
    && locale-gen ko_KR.UTF-8
RUN sudo apt-get upgrade -y\
    && sudo apt-get install vim -y
RUN apt-get install libglib2.0-0 -y

ENV LC_ALL ko_KR.UTF-8
RUN pip install --upgrade pip

# cuda 10.0
# RUN pip install torch==1.2.0 torchvision==0.4.0
# RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# 프로젝트 필요 소스 다운로드
RUN mkdir -p /${fname}
WORKDIR /${fname}
COPY . .

RUN pip install -r requirements.txt
WORKDIR /${fname}/src
RUN ["/bin/bash"]
