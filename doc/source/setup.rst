Getting Started
===============

Compatibility
-------------

This software suite works out-of-the-box on Linux machines. It has not been tested on Windows machines.

Compile from source
-------------------

The source code is publicly available under BSD License. Check it out with ::

  git clone https://github.com/mareikep/calo.git

.. note::
    The project will be available on `github` soon.
    Contact |leaddev| (|leaddevmail|) if you need access now.

Prerequisites
~~~~~~~~~~~~~

* Python 3.5 (or newer)
* Additional python dependencies (listed in ``requirements.txt``)

    .. note::

      You can install all required python dependencies via ::

        pip install -r requirements.txt


Installation
~~~~~~~~~~~~

Generating tools

Run the ``setup.py`` script::

  python setup.py


This will generate a number of executable python scripts, i.e.:

- matcalo
- matcalolearn
- matcalotest

See :doc:`matcalotools` for their respective usage.

Install via PyPi
----------------

Find the project and download files on `PyPi <https://pypi.org/project/calo/>`_ or install directly via `pip` ::

    pip install calo

.. note::
    The project will be available on `PyPi` soon.
    Contact |leaddev| (|leaddevmail|) if you need access now.
