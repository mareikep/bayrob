For Developers
==============


The class :class:`bayrob.core.base.BayRoB` provides the methods :attr:`query_jpts` and :attr:`search_astar`
for querying action models and triggering the *BayRoB* search to generate or refine action sequences.

.. _Learning:

Learning
--------

Learning in *BayRoB* can be facilitated by simply training a :class:`jpt.trees.JPT` model from an existing dataset.
The ``example`` module provides a convenience configuration for updating/retraining the existing models ``move``,
``perception``, ``turn`` and ``pr2``. With minor changes, additional action models can be incorporated into the system.
By running ``python examples/example.py`` a new action model can be trained, or an existing one can be updated. Each action model has
its own folder in ``examples``. The existing models include ``move``, ``perception``, ``turn`` and ``pr2``. There are multiple
general arguments to control the learning process:

* :code:`-v, \-\-verbose {debug,info,warning,error,critical}` sets the verbosity level for the current run
* :code:`-e, \-\-example <model-name>` learns/updates the action model in ``examples/<model-name>``
* :code:`-a, \-\-args <arg> <value>` passes an action-specific argument to the respective ``<model-name>`` module. For accepted arguments, refer to the modules' documentations
* :code:`\-\-recent` will address the most recently generated folder (by a previous run). If not given, a new folder with a timestamp of the current run ``examples/<example-name>/<run-id>`` is created and the respective module's ``generate_data(fp, args)`` function is called to re-generate the data files, the result of which will be stored in ``examples/example-name>/<run-id>/data/000-<model-name>.parquet``.
* :code:`\-\-learn` trains `JPT` models for the ``<model-name>`` action. The model object will be stored in ``examples/example-name>/<run-id>/000-<example-name>.tree``
* :code:`\-\-modulelearn` allows to use a user-specified learning function instead of the default call of ``learn_jpt`` in ``example/example.py``
* :code:`\-\-crossval` triggers the crossvalidation of multiple (predefined) model settings
* :code:`\-\-plot` will trigger plotting of the action model JPT (twice, with and without variable plots). The result will be stored in ``examples/example-name>/<run-id>/plots/000-<example-name>.svg`` and ``examples/example-name>/<run-id>/plots/000-<example-name>-nodist.svg``.
* :code:`\-\-min-samples-leaf <n> and \-\-min-impurity-improvement <n>` passes the ``min_samples_leaf`` or ``min_impurity_improvement`` parameter to the learning function
* :code:`\-\-obstacles` will add obstacle handling in all functions (where necessary)
* :code:`\-\-data` will trigger generating data/world plots by calling the ``examples/<model-name>`` module's ``plot_data`` function
* :code:`\-\-prune` triggers the ``do_prune`` function during the learning of the action model to influence the default behavior

To incorporate a new model into the system, add a new folder ``<model-name>`` in ``examples/`` and provide a
``examples/<model-name>/<model-name>.py`` module implementing the functions ``init``, ``generate_data``, ``learn-jpt``,
``plot_data``, and ``tear_down``, each accepting two arguments: ``fp``, a string representing the filepath to the current
run of the ``<model-namme>`` and ``args``, an argument object allowing to access the arguments passed to the call.

Once a functioning model has been learnt, it can be integrated into the web application.
Place the ``.tree`` file of your model along with its ``.parquet`` data file in a folder ``examples/demo/<model-name>``
which is used by the web application to load the models. The folder structure then looks like the following:

::

    └── examples
        ├── demo
        │   ├── alarm
        │   ├── move
        │   ├── ...
        │   └── <model-name>
        │       ├── 000-<model-name>.tree
        │       └── 000-<model-name>.parquet
        ├── move
        ├── ...
        ├── <model-name>
        │   ├── <YYYY-MM-DD_HH:mm>
        │   │   ├── [crossval] (optional)
        │   │   ├── data
        │   │   │   └── 000-<model-name>.parquet
        │   │   ├── plots
        │   │   │   ├── 000-<model-name>.svg
        │   │   │   ├── 000-<model-name>-nodist.svg
        │   │   │   └── ...
        │   │   └── 000-<model-name>.tree
        │   ├── ...
        │   ├── __init__.py
        │   └── <model-name>.py
        └── ...



.. _Reasoning:

Reasoning
---------

Reasoning in *BayRoB* is performed using the :class:`bayrob.core.base.BayRoB` class, which triggers the query of the
respective action models (of type :class:`jpt.trees.JPT`) with a passed :class:`bayrob.core.base.Query` object. This
object requires information about made observations (:attr:`evidence`) and the variables (:attr:`queryvars`) one is
interested in.

    :Example:

    >>> import os
    >>> from jpt.base.intervals import ContinuousSet
    >>> from bayrob.core.base import BayRoB, Query
    >>> from bayrob.utils import locs
    >>>
    >>> preset = {
    ...     "evidence": {
    ...         'detected(milk)': False,
    ...         'x_in': ContinuousSet(58, 68),
    ...         'y_in': ContinuousSet(70, 80),
    ...         'nearest_furniture': 'fridge'
    ...     },
    ...     "queryvars": ['daytime', 'open(fridge_door)']
    ... }
    >>>
    >>> bayrob = BayRoB()
    >>> bayrob.adddatapath([os.path.join(locs.examples, 'demo', "perception")])
    >>> allvars_ = {v.name: v for v in bayrob.models['perception'].variables}
    >>>
    >>> qo = Query()
    >>> qo.model = bayrob.models['perception']
    >>> qo.evidence = {allvars_[k]: v for k, v in preset['evidence'].items()}
    >>> qo.queryvars = [bayrob.models['perception'].varnames[k] for k in preset['queryvars']]
    >>>
    >>> bayrob.query = qo
    >>> bayrob.query_jpts()
    >>> cond, post = bayrob.result.result
    >>> print(cond)
    <JPT #innernodes = 9, #leaves = 7 (16 total)>
    >>> print(post)
    <VariableMap {x_in: <jpt.distributions.univariate.numeric.Numeric object at 0x79b4c4a50c40>, y_in: <jpt.distributions.univariate.numeric.Numeric object at 0x79b4c4bf6190>, xdir_in: <jpt.distributions.univariate.numeric.Numeric object at 0x79b4c4bf8190>, ydir_in: <jpt.distributions.univariate.numeric.Numeric object at 0x79b4c4bf22b0>, daytime: <DAYTIME_TYPE_S p=[morning=0.314;post-breakfast=0.133;night=0.309;lunchtime=0.056;[...]

The result of a *BayRoB* query is a tuple ``(cond, post)`` where ``cond`` is the conditional tree
(:class:`jpt.trees.JPT`) and ``post`` is a mapping of the variables (:class:`jpt.variables.Variable`) specified in
:attr:`queryvars` to their respective posterior distributions (:class:`jpt.distributions.univariate.numeric.Numeric`,
:class:`jpt.distributions.univariate.multinomial.Bool`, :class:`jpt.distributions.univariate.multinomial.Multinomial`,
...):

.. _Plan Refinement:

Plan Refinement
---------------

*BayRoB* can be used to refine robot plans by searching for a path from the current state to a desired, user-defined
goal state. The result is a sequence of belief states representing the prospected outcomes of parameterized action steps
that, when executed in order, will most likely lead to the desired goal state. The class :class:`bayrob.core.base.Search`
encapsulates the information required to perform the search, i.e. the initial (belief) state of the agent and a goal
specification, each augmented by a certain tolerance, and the direction, the search is supposed to be performed in.
The search specification is then passed to the :class:`bayrob.core.base.BayRoB`, which triggers the search algorithm.

    :Example:

    >>> import os
    >>> from jpt.base.intervals import ContinuousSet
    >>> from bayrob.core.base import BayRoB, Search
    >>> from bayrob.utils import locs
    >>> preset = {
    ...     "init": {
    ...         'x_in': 62,
    ...         'y_in': 74,
    ...         'xdir_in': .3,
    ...         'ydir_in': .9,
    ...     },
    ...     "init_tolerances": {
    ...         'x_in': .1,
    ...         'y_in': .1,
    ...         'xdir_in': .01,
    ...         'ydir_in': .01,
    ...     },
    ...     "goal": {
    ...         'detected(milk)': {True},
    ...     },
    ...     "goal_tolerances": {},
    ...     "bwd": True
    ... }
    >>>
    >>> bayrob = BayRoB()
    >>> bayrob.adddatapath([os.path.join(locs.examples, 'demo', d) for d in os.listdir(os.path.join(locs.examples, 'demo'))])
    >>> allvars = bayrob.models['move'].variables + \
                  bayrob.models['turn'].variables + \
                  bayrob.models['perception'].variables
    >>> allvars_ = {v.name: v for v in allvars}
    >>>
    >>> asr = Search()
    >>> asr.bwd = preset['bwd']
    >>> asr.init = {allvars_[k]: v for k, v in preset['init'].items()}
    >>> asr.init_tolerances = {allvars_[k]: v for k, v in preset['init_tolerances'].items()}
    >>> asr.goal = {allvars_[k]: v for k, v in preset['goal'].items()}
    >>> asr.goal_tolerances = {allvars_[k]: v for k, v in preset['goal_tolerances'].items()}
    >>>
    >>> bayrob.query = asr
    >>> bayrob.search_astar()
    >>> seq = bayrob.result.result
    >>> print(seq)
    [State[x_in: [59.45,65.56[;y_in: [73.71,74.06[;xdir_in: [0.35,0.40[;ydir_in: [0.91,0.93[;collided: {False};daytime: {'post-breakfast'};open(fridge_door): {True};open(cupboard_door_left): {False};open(cupboard_door_right): {False};open(kitchen_unit_drawer): {False};open(stove_door): {False};detected(cup): {False};detected(cutlery): {False};detected(bowl): {False};detected(sink): {False};detected(milk): {True};detected(beer): {True};detected(cereal): {False};detected(stovetop): {False};detected(pot): {False};nearest_furniture: [...]

The result of a *BayRoB* search is a sequence of belief states (:class:`bayrob.core.astar_jpt.State`) represented by
instances of the probability distribution classes :class:`jpt.distributions.univariate.numeric.Numeric`,
:class:`jpt.distributions.univariate.multinomial.Bool`, and :class:`jpt.distributions.univariate.multinomial.Multinomial`,
for multiple variables (:class:`jpt.variables.Variable`).
