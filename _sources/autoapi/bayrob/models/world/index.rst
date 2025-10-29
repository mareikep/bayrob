bayrob.models.world
===================

.. py:module:: bayrob.models.world


Classes
-------

.. autoapisummary::

   bayrob.models.world.World
   bayrob.models.world.Agent
   bayrob.models.world.Grid
   bayrob.models.world.GridAgent


Module Contents
---------------

.. py:class:: World

.. py:class:: Agent(world: World = None)

.. py:class:: Grid(x=None, y=None)

   Bases: :py:obj:`World`


   .. py:attribute:: coords


   .. py:method:: obstacle(x0, y0, x1, y1, name: str = None) -> None


   .. py:property:: obstacles
      :type: List[List[float]]



   .. py:property:: obstaclenames
      :type: List[List[float]]



   .. py:method:: collides_obstacle(pos: Tuple[float, float]) -> bool


   .. py:method:: collides_wall(pos: Tuple[float, float]) -> bool


   .. py:method:: collides(pos: Tuple[float, float]) -> bool


.. py:class:: GridAgent(world: Grid = None, pos: Tuple[float, float] = None, dir: Tuple[float, float] = None)

   Bases: :py:obj:`Agent`


   .. py:property:: pos
      :type: (float, float)



   .. py:property:: dir
      :type: (float, float)



   .. py:property:: x
      :type: float



   .. py:property:: y
      :type: float



   .. py:property:: dirx
      :type: float



   .. py:property:: diry
      :type: float



   .. py:property:: world
      :type: Grid



   .. py:property:: collided
      :type: bool



   .. py:method:: init_random()


