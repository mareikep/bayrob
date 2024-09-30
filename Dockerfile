FROM ubuntu:22.04

# install software and dependencies for running python app
RUN apt update --fix-missing && \
    apt install -y  \
    apt-utils  \
    wget  \
    git \
    gcc \
    python-is-python3  \
    software-properties-common  && \
    add-apt-repository -y ppa:deadsnakes && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends  \
    python3.8-dev \
    build-essential \
    python3.8-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    :

# create virtual environment for python 3.8 (required for pyrap)
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# copy source code and install requirements for app and docs
COPY . /bayrob-dev
RUN mkdir /bayrob-dev/logs
RUN cd bayrob-dev && python -m pip install "Cython<=0.29.35" numpy && python -m pip install -U -r requirements.txt
#RUN cd bayrob-dev/docs && python -m pip install -U -r requirements.txt && make clean && make html
ENV PYTHONPATH=/bayrob-dev/src
