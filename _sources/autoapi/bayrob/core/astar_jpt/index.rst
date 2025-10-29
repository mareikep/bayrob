bayrob.core.astar_jpt
=====================

.. py:module:: bayrob.core.astar_jpt


Attributes
----------

.. autoapisummary::

   bayrob.core.astar_jpt.logger


Classes
-------

.. autoapisummary::

   bayrob.core.astar_jpt.State
   bayrob.core.astar_jpt.Goal
   bayrob.core.astar_jpt.SubAStar
   bayrob.core.astar_jpt.SubAStarBW


Module Contents
---------------

.. py:data:: logger

.. py:class:: State(d: dict = None)

   Bases: :py:obj:`dict`


   dict() -> new empty dictionary
   dict(mapping) -> new dictionary initialized from a mapping object's
       (key, value) pairs
   dict(iterable) -> new dictionary initialized as if via:
       d = {}
       for k, v in iterable:
           d[k] = v
   dict(**kwargs) -> new dictionary initialized with the name=value pairs
       in the keyword argument list.  For example:  dict(one=1, two=2)


   .. py:attribute:: leaf
      :value: None



   .. py:attribute:: tree
      :value: None



   .. py:method:: similarity(other: State) -> float


   .. py:method:: distance(other, vars=None)


.. py:class:: Goal(d: dict = None)

   Bases: :py:obj:`State`


   dict() -> new empty dictionary
   dict(mapping) -> new dictionary initialized from a mapping object's
       (key, value) pairs
   dict(iterable) -> new dictionary initialized as if via:
       d = {}
       for k, v in iterable:
           d[k] = v
   dict(**kwargs) -> new dictionary initialized with the name=value pairs
       in the keyword argument list.  For example:  dict(one=1, two=2)


   .. py:method:: similarity(other: Goal) -> float


   .. py:method:: distance(other, vars=None)


.. py:class:: SubAStar(initstate: Any, goal: Any, models: Dict, state_similarity: float = 0.2, goal_confidence: float = 0.01, n_workers: int = None, use_multiprocessing: bool = True)

   Bases: :py:obj:`bayrob.core.astar.AStar`


   .. py:attribute:: models


   .. py:attribute:: state_t


   .. py:attribute:: goal_t


   .. py:attribute:: use_multiprocessing


   .. py:attribute:: n_workers


   .. py:attribute:: pool
      :value: None



   .. py:attribute:: timeout
      :value: 60



   .. py:attribute:: cache
      :value: None



   .. py:attribute:: fast_jaccard_cont
      :value: None



   .. py:attribute:: fast_jaccard_set
      :value: None



   .. py:method:: init()


   .. py:method:: jaccard_similarity(d1: jpt.base.intervals.ContinuousSet, d2: jpt.base.intervals.ContinuousSet) -> float
      :staticmethod:



   .. py:method:: isgoal(node: bayrob.core.astar.Node) -> bool


   .. py:method:: generate_steps(node) -> List[Any]

      Generates potential next steps (parallelized with ProcessPoolExecutor).
              



   .. py:method:: generate_successors(node) -> List[bayrob.core.astar.Node]


.. py:class:: SubAStarBW(initstate: State, goal: Goal, models: Dict, state_similarity: float = 0.2, goal_confidence: float = 0.01, n_workers: int = None, use_multiprocessing: bool = True)

   Bases: :py:obj:`SubAStar`


   .. py:method:: init()


   .. py:method:: jaccard_similarity(d1: Union[jpt.base.intervals.ContinuousSet, set], d2: Union[jpt.base.intervals.ContinuousSet, set]) -> float
      :staticmethod:



   .. py:method:: isgoal(node: bayrob.core.astar.Node) -> bool


   .. py:method:: get_ancestor(node)
      :staticmethod:



   .. py:method:: reverse(t: jpt.trees.JPT, node: bayrob.core.astar.Node, treename: str = None) -> List

      Serial implementation of reverse (kept for compatibility).
              



   .. py:method:: generate_steps(node: bayrob.core.astar.Node) -> List[Any]

      Generates potential previous steps (parallelized with ProcessPoolExecutor).
              



   .. py:method:: generate_successors(node: bayrob.core.astar.Node) -> List[bayrob.core.astar.Node]


