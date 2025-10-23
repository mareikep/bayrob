Reasoning
=====

Reasoning in #bayrob is performed using the :py:class:`bayrob.core.base.BayRoB` class, which triggers the query of the
respective action models (of type :py:class:`jpt.trees.JPT`) with a passed :py:class:`bayrob.core.base.Query` object. This
object requires information about made observations (`evidence`) and the variables (`queryvars`) one is
interested in.

.. code-block:: py

    import os
    from jpt.base.intervals import ContinuousSet
    from bayrob.core.base import BayRoB, Query
    from bayrob.utils import locs

    preset = {
         "evidence": {
             'detected(milk)': False,
             'x_in': ContinuousSet(58, 68),
             'y_in': ContinuousSet(70, 80),
             'nearest_furniture': 'fridge'
         },
         "queryvars": ['daytime', 'open(fridge_door)']
     }

    bayrob = BayRoB()
    bayrob.adddatapath([os.path.join(locs.examples, 'demo', "perception")])
    allvars_ = {v.name: v for v in bayrob.models['perception'].variables}

    qo = Query()
    qo.model = bayrob.models['perception']
    qo.evidence = {allvars_[k]: v for k, v in preset['evidence'].items()}
    qo.queryvars = [bayrob.models['perception'].varnames[k] for k in preset['queryvars']]

    bayrob.query = qo
    bayrob.query_jpts()
    cond, post = bayrob.result.result
    print(cond)
    print(post)

The result of a *BayRoB* query is a tuple ``(cond, post)`` where ``cond`` is the conditional tree
(:py:class:`jpt.trees.JPT`) and ``post`` is a mapping of the variables (:py:class:`jpt.variables.Variable`) specified in
`queryvars` to their respective posterior distributions (:py:class:`jpt.distributions.univariate.numeric.Numeric`,
:py:class:`jpt.distributions.univariate.multinomial.Bool`, :py:class:`jpt.distributions.univariate.multinomial.Multinomial`,
...):