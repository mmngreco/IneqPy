FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y \
    build-essential \
    python3.6 \
    python3.6-dev \
    python3-pip \
    python3-rtree \
    python3.6-venv \
    gnuplot \ 
    git \
    vim
RUN python3.6 -m pip install pip -U
RUN python3.6 -m pip install wheel -U
RUN mkdir -p /root/git
WORKDIR /root/git

RUN python3.6 -m venv venv
# RUN git clone https://github.com/mmngreco/ineqpy
# https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
CMD source venv/bin/activate && /bin/bash
