bayrob.core.base
================

.. py:module:: bayrob.core.base


Attributes
----------

.. autoapisummary::

   bayrob.core.base.logger
   bayrob.core.base.jsonlogger


Classes
-------

.. autoapisummary::

   bayrob.core.base.Search
   bayrob.core.base.Query
   bayrob.core.base.BayRoB


Module Contents
---------------

.. py:data:: logger

.. py:data:: jsonlogger

.. py:class:: Search

   The Search object encapsulating required information for a `BayRoB` plan refinement.


   .. py:attribute:: goal


   .. py:attribute:: goal_tolerances


   .. py:attribute:: init


   .. py:attribute:: init_tolerances


   .. py:attribute:: bwd
      :value: True



.. py:class:: Query

   .. py:attribute:: evidence


   .. py:attribute:: model
      :value: None



   .. py:attribute:: modelname
      :value: None



   .. py:attribute:: querystr
      :value: ''



   .. py:attribute:: queryvars
      :value: []



   .. py:attribute:: plottype
      :value: None



   .. py:attribute:: plot_tree
      :value: False



.. py:class:: BayRoB(query: Union[Query, Search] = None, datapaths=None)

   The BayRoB reasoning system.


   .. py:attribute:: omitmodels
      :value: []



   .. py:attribute:: search_mode
      :value: 'reverse'



   .. py:class:: Result(query_object, success=False, error=None, message=None, result=None)

      The result of either query or plan refinement in `BayRoB`. The object contains information about the success
      of the operation and provides information about possible errors or additional messages relevant to interpret
      the outcome.


      .. py:property:: query_object
         :type: Union[Search, Query]



      .. py:property:: success
         :type: bool


         Returns True if the task was executed without errors, otherwise False.



      .. py:property:: error
         :type: str


         Returns True if the task was executed without errors, otherwise False.



      .. py:property:: message
         :type: str


         A message with additional information possibly relevant to interpret the result.



      .. py:property:: result
         :type: Union[Tuple[jpt.trees.JPT, jpt.variables.VariableMap], List[bayrob.core.astar_jpt.State]]


         The result of the task execution. Either a sequence of actions (plan refinement) or a tuple
         `(cond, post)` of a conditional tree and posterior distributions after reasoning over an individual model.
          



      .. py:method:: clear()

         Remove all information of the result except the passed query object. 




   .. py:method:: adddatapath(path) -> None

      Adds a path to an action model containing the model file containing an :class:`jpt.trees.JPT` instance and
      a .parquet file containing the original data to train the model. After adding the path(s), the model is loaded.
      The model objects as well as the path to the datafiles are accessible using the attributes :attr:`models` and
      :attr:`datasets`.

      :param path:    The path(s) to an action model containing data and model files
      :type path:     str or iterable



   .. py:method:: removedatapath(path) -> None

      Removes one or more path(s) pointing to model and data files. The respective models are removed from the
      :attr:`models`.

      :param path:    The path(s) to an action model containing data and model files
      :type path:     str or iterable



   .. py:property:: query
      :type: Union[Query, Search]


      The :attr:`query` attribute is an instance of either :class:`Query` or :class:`Search` specifying the
      required information for reasoning over an individual model or refining a plan.



   .. py:property:: datapaths
      :type: list


      The datapaths containing the paths action models.

      :returns:   list of paths to the folders containing pickled :class:`jpt.trees.JPT` models and the .parquet data
                  file used for training the model
      :rtype:     List[str]



   .. py:property:: models
      :type: Dict[str, jpt.trees.JPT]


      The action models used for reasoning.

      :returns: a mapping from model names to :class:`jpt.trees.JPT` models
      :rtype: Dict[str, JPT]



   .. py:property:: datasets
      :type: Dict[str, str]


      The datasets containing the data used for training the models.

      :returns: a mapping from model names to the path of the .parquet data file used for training the model
      :rtype: Dict[str, str]



   .. py:property:: result

      The result object to be processed by the calling entity (e.g. to visualize).

      :returns: the generated tree representation of the inference result
      :rtype: :class:`core.base.BayRoB.Result`



   .. py:method:: search_astar() -> None

      This function performs a plan refinement by searching a path from the initial state to the goal state
      defined in the :class:`bayrob.core.base.Search` object passed to the class beforehand.



   .. py:method:: query_jpts() -> None

      This function performs a query to an action model as specified in the :class:`bayrob.core.base.Query`
      object passed to the class beforehand.



