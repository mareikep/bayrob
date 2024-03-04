
API-Specification
=================

|project| comes with an easy-to-use API, which lets you use the modules provided by |project| of which each solves an
own inference task conveniently in your own applications.

****

|project|
*********

.. automodule:: bayrob.core.base
    :members: BayRoB

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

****

.. ..


****

Configuration
*************

.. class:: bayrob.config.Config
.. automodule:: bayrob.config