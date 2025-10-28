Plan Refinement
===============

*BayRoB* can be used to refine robot plans by searching for a path from the current state to a desired, user-defined
goal state. The result is a sequence of belief states representing the prospected outcomes of parameterized action steps
that, when executed in order, will most likely lead to the desired goal state. The class :py:class:`bayrob.core.base.Search`
encapsulates the information required to perform the search, i.e. the initial (belief) state of the agent and a goal
specification, each augmented by a certain tolerance, and the direction, the search is supposed to be performed in.
The search specification is then passed to the :py:class:`bayrob.core.base.BayRoB`, which triggers the search algorithm.

.. code-block:: py

    import os
    from jpt.base.intervals import ContinuousSet
    from bayrob.core.base import BayRoB, Search
    from bayrob.utils import locs
    preset = {
         "init": {
             'x_in': 62,
             'y_in': 74,
             'xdir_in': .3,
             'ydir_in': .9,
         },
         "init_tolerances": {
             'x_in': .1,
             'y_in': .1,
             'xdir_in': .01,
             'ydir_in': .01,
         },
         "goal": {
             'detected(milk)': {True},
         },
         "goal_tolerances": {},
         "bwd": True
     }

    bayrob = BayRoB()
    bayrob.adddatapath([os.path.join(locs.examples, 'demo', d) for d in os.listdir(os.path.join(locs.examples, 'demo'))])
    allvars = bayrob.models['move'].variables + \
                      bayrob.models['turn'].variables + \
                      bayrob.models['perception'].variables
    allvars_ = {v.name: v for v in allvars}

    asr = Search()
    asr.bwd = preset['bwd']
    asr.init = {allvars_[k]: v for k, v in preset['init'].items()}
    asr.init_tolerances = {allvars_[k]: v for k, v in preset['init_tolerances'].items()}
    asr.goal = {allvars_[k]: v for k, v in preset['goal'].items()}
    asr.goal_tolerances = {allvars_[k]: v for k, v in preset['goal_tolerances'].items()}

    bayrob.query = asr
    bayrob.search_astar()
    seq = bayrob.result.result
    print(seq)


The result of a *BayRoB* search is a sequence of belief states (:py:class:`bayrob.core.astar_jpt.State`) represented by
instances of the probability distribution classes :py:class:`jpt.distributions.univariate.numeric.Numeric`,
:py:class:`jpt.distributions.univariate.multinomial.Bool`, and :py:class:`jpt.distributions.univariate.multinomial.Multinomial`,
for multiple variables (:py:class:`jpt.variables.Variable`).
