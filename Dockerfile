FROM ubuntu:22.04

# install software and dependencies for running python app
RUN apt update --fix-missing && \
    apt install -y  \
    apt-utils  \
    wget  \
    git \
    gcc \
    parallel \
    python-is-python3  \
    software-properties-common  && \
    add-apt-repository -y ppa:deadsnakes && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends  \
    python3.8-dev \
    build-essential \
    python3.8-venv && \
    wget https://github.com/casey/just/releases/download/1.42.4/just-1.42.4-x86_64-unknown-linux-musl.tar.gz && \
    tar -xzf just-1.42.4-x86_64-unknown-linux-musl.tar.gz && \
    mv just /usr/local/bin/ && \
    rm just-1.42.4-x86_64-unknown-linux-musl.tar.gz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    :

# create virtual environment for python 3.8 (required for pyrap)
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# copy source code and install requirements for app and docs
COPY . /bayrob-dev

# create logs folder
RUN mkdir -p /bayrob-dev/logs

# install dependencies for bayrob-dev
WORKDIR /bayrob-dev
RUN python -m pip install "Cython<=0.29.35" numpy \
    && python -m pip install -U -r requirements.txt

# build docs with just
WORKDIR /bayrob-dev/docs
RUN just compile-all

# default working directory for container
WORKDIR /bayrob-dev/src/bayrob/web

ENV PYTHONPATH=/bayrob-dev/src

