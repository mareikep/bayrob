Learning
========

Learning in #bayrob can be facilitated by simply training a :py:class:`JPT <jpt.trees.JPT>` model from an existing dataset.
The ``example`` module provides a convenience configuration for updating/retraining the existing models ``move``,
``perception``, ``turn`` and ``pr2``. With minor changes, additional action models can be incorporated into the system.
By running ``python examples/example.py`` a new action model can be trained, or an existing one can be updated. Each action model has
its own folder in ``examples``. The existing models include ``move``, ``perception``, ``turn`` and ``pr2``. There are multiple
general arguments to control the learning process:

- `-v, --verbose {debug,info,warning,error,critical}` sets the verbosity level for the current run
- `-e, --example <model-name>` learns/updates the action model in ``examples/<model-name>``
- `-a, --args <arg> <value>` passes an action-specific argument to the respective ``<model-name>`` module. For accepted arguments, refer to the modules' documentations
- `--recent` will address the most recently generated folder (by a previous run). If not given, a new folder with a timestamp of the current run ``examples/<example-name>/<run-id>`` is created and the respective module's ``generate_data(fp, args)`` function is called to re-generate the data files, the result of which will be stored in ``examples/example-name>/<run-id>/data/000-<model-name>.parquet``.
- `--learn` trains `JPT` models for the ``<model-name>`` action. The model object will be stored in ``examples/example-name>/<run-id>/000-<example-name>.tree``
- `--modulelearn` allows to use a user-specified learning function instead of the default call of ``learn_jpt`` in ``example/example.py``
- `--crossval` triggers the crossvalidation of multiple (predefined) model settings
- `--plot` will trigger plotting of the action model JPT (twice, with and without variable plots). The result will be stored in ``examples/example-name>/<run-id>/plots/000-<example-name>.svg`` and ``examples/example-name>/<run-id>/plots/000-<example-name>-nodist.svg``.
- `--min-samples-leaf <n> and --min-impurity-improvement <n>` passes the ``min_samples_leaf`` or ``min_impurity_improvement`` parameter to the learning function
- `--obstacles` will add obstacle handling in all functions (where necessary)
- `--data` will trigger generating data/world plots by calling the ``examples/<model-name>`` module's ``plot_data`` function
- `--prune` triggers the ``do_prune`` function during the learning of the action model to influence the default behavior

To incorporate a new model into the system, add a new folder ``<model-name>`` in ``examples/`` and provide a
``examples/<model-name>/<model-name>.py`` module implementing the functions ``init``, ``generate_data``, ``learn-jpt``,
``plot_data``, and ``tear_down``, each accepting two arguments: ``fp``, a string representing the filepath to the current
run of the ``<model-name>`` and ``args``, an argument object allowing to access the arguments passed to the call.

Once a functioning model has been learnt, it can be integrated into the web application.
Place the ``.tree`` file of your model along with its ``.parquet`` data file in a folder ``examples/demo/<model-name>``
which is used by the web application to load the models. The folder structure then looks like the following:

.. code-block:: bash

        └── examples/
            ├── demo/
            │   ├── alarm
            │   ├── move
            │   ├── ...
            │   └── <model-name>/
            │       ├── 000-<model-name>.tree
            │       └── 000-<model-name>.parquet
            ├── move/
            ├── ...
            ├── <model-name>/
            │   ├── <YYYY-MM-DD_HH:mm>/
            │   │   ├── [crossval/] (optional)
            │   │   ├── data/
            │   │   │   └── 000-<model-name>.parquet
            │   │   ├── plots/
            │   │   │   ├── 000-<model-name>.svg
            │   │   │   ├── 000-<model-name>-nodist.svg
            │   │   │   └── ...
            │   │   └── 000-<model-name>.tree
            │   ├── ...
            │   ├── __init__.py
            │   └── <model-name>.py
            └── ...
