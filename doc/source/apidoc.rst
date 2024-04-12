
API-Specification
=================

|project| comes with an easy-to-use API, which lets you use the modules provided by |project| of which each solves an
own inference task conveniently in your own applications.

****

|project|
*********

.. automodule:: bayrob.core.base
    :members:
    :exclude-members: BayRoB, Search, Query

    .. autoclass:: Query
        :members:

    .. autoclass:: Search
        :members:

    .. autoclass:: BayRoB
        :members:
        :exclude-members: Result

        .. autoclass:: bayrob.core.base.BayRoB::Result
            :members:

.. automodule:: bayrob.core.astar
    :members: AStar, BiDirAStar, Node

.. automodule:: bayrob.core.astar_jpt
    :members: Goal, State, SubAStar, SubAStarBW

****

Algorithms
**********

.. automodule:: jpt.trees
    :members: JPT


****

Utils
*****

.. automodule:: jpt.base.intervals
    :members: ContinuousSet

.. automodule:: jpt.variables
    :members: Variable, NumericVariable, SymbolicVariable, VariableMap

