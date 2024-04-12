FROM ubuntu:22.04

# install software and dependencies for running python app
RUN apt update --fix-missing && \
    apt install -y apt-utils wget python-is-python3 software-properties-common build-essential && \
    add-apt-repository -y ppa:deadsnakes && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3.8-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    :

# create virtual environment for python 3.8 (required for pyrap)
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# copy source code and install requirements for app and docs
COPY . /bayrob-dev
RUN mkdir /bayrob-dev/logs
RUN cd bayrob-dev && python -m pip install -U -r requirements.txt
RUN cd bayrob-dev/doc && python -m pip install -U -r requirements.txt
ENV PYTHONPATH=/bayrob-dev/src:$PYTHONPATH

# uncomment the following two lines and run `sudo docker run --rm -ti -p 5005:5005 bayrob-web-img` (make run) or
# comment them out and run `sudo docker compose up` (make compose),
# both after running `sudo docker build --tag bayrob-web-img .` (make build).
#EXPOSE 5005
#ENTRYPOINT ["python3", "/bayrob-dev/src/bayrob/web/server.py", "-p", "5005", "-i", "0.0.0.0"]