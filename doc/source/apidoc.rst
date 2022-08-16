
API-Specification
=================

|project| comes with an easy-to-use API, which lets you use the modules provided by |project| of which each solves an
own inference task conveniently in your own applications.

****

|project|
*********

.. automodule:: calo.core.base
    :members: CALO, Hypothesis, ResTree, Step

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

****

Example
*******

.. automodule:: jpt.variables
    :members: Variable, NumericVariable, SymbolicVariable, VariableMap

****

.. ..

 <!--- ATTENTION! The docs for the Database and Configuration singletons are generated differently because
 sphinx cannot compile it properly due to the classname overriding. While configuration returns the direct class
 instantiation of CALOConfig and therefore allows this slightly hacky class/automodule generation, the Database
 connection returns mongodb, which is a class attribute and therefore ignores any kind of docstring in that module.
 This is why we put the documentation here.
 --->

Database
********

.. autoclass:: calo.database.connection.DBConnection

The Database Connection singleton.

If the mongo db connection settings are defined in the default config file, these settings are used to set up
the db connection. If not, |project| connects to the local default database.

.. seealso:: :class:`calo.config.Config`


****

Configuration
*************

.. class:: calo.config.Config
.. automodule:: calo.config