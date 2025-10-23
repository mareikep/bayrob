# Getting Started

The *BayRoB* system can be executed in multiple ways. To try out the web app locally, just follow the instructions
in [Use Web App in Docker](#use-web-app-in-docker) to start the application in a docker container. To inspect and use the code in your
own project, follow the instructions in [Compile from Source](#compile-from-source). To generate and run your own docker container after
updating the code (e.g. after adding more action models), see [Generate Docker Container](#generate-docker-container). 

****

## Compatibility

This software should work on most Linux distributions. It has not been tested on Windows machines.

****

## Use Web App in Docker

Create a file `docker-compose.yaml` with the following contents

    version: '3'

    services:
      bayrobweb:
        image: "bayrob-web-img"
        container_name: bayrob-web-container
        working_dir: /bayrob-dev/src/bayrob/web
        command: "python3 server.py -p 5005 -i 0.0.0.0"
        environment:
          - "PYTHONUNBUFFERED:1"
        ports:
          - "5005:5005"

and (pull and) start the container using the command: 

    $ sudo docker compose up

Open the following address in a web browser: 

    http://127.0.0.1:5005/bayrob/

See [For Users](for_users.md) for instructions on how to use the web application.

****

## Compile from Source

The source code is publicly available under BSD License. Check it out with 

    $ git clone https://github.com/mareikep/bayrob.git

### Prerequisites

* Python >= 3.8
* Additional python dependencies (listed in ``requirements.txt``)

> **_NOTE:_**
        Amongst others, the system requires ``pyjpt`` and ``pyrap``, which are included in the ``3rdparty`` folder as
        precompiled wheels. While ``pyjpt`` is essential in *BayRoB* since it provides the base functionality for
        probabilistic reasoning, ``pyrap`` is only required for the use of the web application.
        You can install all required python dependencies (including the two precompiled wheels) via ``pip``:
                
        
    $ pip install -r requirements.txt

If a newer version of ``pyjpt`` is required, since the framework is in constant development, update it via ``pip``
or download its source code from [GitHub.](https://github.com/joint-probability-trees/jpt-dev)


### Usage

The module :class:`bayrob.core.base.BayRoB` is the entry point for using the system. The two methods
`.search_astar()` and `.query_jpts()` allow querying individual models or run the search algorithm to refine
action plans. See sections :ref:`Reasoning` and :ref:`Plan Refinement` in [For Developers](for_developers.md) for examples and the API
for additional information on the module's functions.

****

## Generate Docker Container

If additional models were integrated into the system (following the instructions in [Learning](for_developers.md)), the Docker
container needs to be rebuilt to allow querying them in the (local) web application. A ``Dockerfile`` and a
``docker-compose.yaml`` file are provided with the source code and can be found on root level of the git checkout.
The ``Makefile`` contains the commands to build and compose the container. Run 

    $ sudo docker build --tag bayrob-web-img .

or

    $ make build

in the root directory of the system and

    $ sudo docker compose up

or

    $ make compose

to start it. As described above, the webapp can then be accessed via web browser using the address 

    http://127.0.0.1:5005/bayrob/

See [For Users](for_users.md) for more information and a user manual of the web app. The added models should now appear in the
dropdown field of the `BayRoB Query` window. If they are not shown, check, if you placed the files in the correct
folder. Note, that only files in subfolders of `examples/demo` will be loaded in the web application.

****
