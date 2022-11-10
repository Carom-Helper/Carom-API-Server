FROM 31f3b6f28041

SHELL ["/bin/bash", "-c"]

# Setup user account
ARG fname
ARG portnum

# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive
ENV Portnum="0.0.0.0:${portnum}"

# install
RUN apt-get update && apt-get install -y sudo \
    && apt-get install -y libgl1-mesa-glx git locales \
    && locale-gen ko_KR.UTF-8

ENV LC_ALL ko_KR.UTF-8
RUN pip install --upgrade pip

# RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# 프로젝트 필요 소스 다운로드
RUN mkdir -p /${fname}
WORKDIR /${fname}
COPY . .

RUN pip install -r requirements.txt
WORKDIR /${fname}/src
CMD python manage.py runserver ${Portnum}
