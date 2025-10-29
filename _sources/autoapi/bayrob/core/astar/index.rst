bayrob.core.astar
=================

.. py:module:: bayrob.core.astar


Attributes
----------

.. autoapisummary::

   bayrob.core.astar.logger


Classes
-------

.. autoapisummary::

   bayrob.core.astar.Node
   bayrob.core.astar.AStar
   bayrob.core.astar.BiDirAStar


Module Contents
---------------

.. py:data:: logger

.. py:class:: Node(state: Any, g: float, h: float, parent: Node = None)

   Abstract Node class for abstract A* Algorithm
       


   .. py:attribute:: g


   .. py:attribute:: h


   .. py:attribute:: state


   .. py:attribute:: parent
      :value: None



   .. py:property:: f
      :type: float



.. py:class:: AStar(initstate: Any, goal: Any, state_similarity: float = 0.9, goal_confidence: float = 0.01, **kwargs)

   Abstract A* class. Inheriting classes need to implement functions for
   goal check, path retraction and successor generation.


   .. py:attribute:: initstate


   .. py:attribute:: goal


   .. py:attribute:: open
      :value: []



   .. py:attribute:: closed
      :value: []



   .. py:attribute:: verbose


   .. py:attribute:: plotme
      :value: False



   .. py:attribute:: reached
      :value: False



   .. py:method:: init()
      :abstractmethod:



   .. py:property:: state_similarity
      :type: float



   .. py:property:: goal_confidence
      :type: float



   .. py:method:: h(state: Any) -> float
      :abstractmethod:



   .. py:method:: stepcost(state: Any, parent: Any) -> float
      :abstractmethod:



   .. py:method:: generate_successors(node: Node) -> List[Node]
      :abstractmethod:



   .. py:method:: isgoal(node: Node) -> bool
      :abstractmethod:


      Check if current node is goal node



   .. py:method:: retrace_path(node) -> Any


   .. py:method:: search() -> Any


   .. py:method:: plot(node, **kwargs) -> Any
      :abstractmethod:



.. py:class:: BiDirAStar(f_astar: type, b_astar: type, initstate: Any, goal: Any, state_similarity: float = 0.9, goal_confidence: float = 0.01, **kwargs)

   .. py:attribute:: state_t


   .. py:attribute:: goal_t


   .. py:attribute:: initstate


   .. py:attribute:: goal


   .. py:attribute:: f_astar


   .. py:attribute:: b_astar


   .. py:attribute:: reached
      :value: False



   .. py:property:: state_similarity
      :type: float



   .. py:property:: goal_confidence
      :type: float



   .. py:method:: retrace_path(fnode: Node, bnode: Node) -> List


   .. py:method:: common_node(fnode: Node, bnode: Node) -> bool


   .. py:method:: search() -> Any


