bayrob.application.astar_jpt_app
================================

.. py:module:: bayrob.application.astar_jpt_app


Classes
-------

.. autoapisummary::

   bayrob.application.astar_jpt_app.SubAStar_
   bayrob.application.astar_jpt_app.SubAStarBW_


Module Contents
---------------

.. py:class:: SubAStar_(initstate: bayrob.core.astar_jpt.State, goal: bayrob.core.astar_jpt.Goal, models: Dict, state_similarity: float = 0.2, goal_confidence: float = 0.2)

   Bases: :py:obj:`bayrob.core.astar_jpt.SubAStar`


   .. py:method:: stepcost(state, parent) -> float


   .. py:method:: h(state: bayrob.core.astar_jpt.State) -> float


   .. py:method:: plot_pos(path: List, title: str = None, conf: float = None, limx: Tuple = None, limy: Tuple = None, limz: Tuple = None, save: str = None, show: bool = True) -> plotly.graph_objs.Figure


   .. py:method:: plot_dir(path: List, title: str = None, conf: float = None, limx: Tuple = None, limy: Tuple = None, limz: Tuple = None, save: str = None, show: bool = True) -> plotly.graph_objs.Figure


   .. py:method:: plot_path(xvar, yvar, p: List, obstacles: List = None, title: str = None, save: str = None, show: bool = False, limx: Tuple = None, limy: Tuple = None) -> plotly.graph_objs.Figure


   .. py:method:: gendata(xvar, yvar, state, conf: float = None)


   .. py:method:: plot(node: jpt.trees.Node, **kwargs) -> plotly.graph_objs.Figure

      ONLY FOR GRIDWORLD DATA
              



.. py:class:: SubAStarBW_(initstate: bayrob.core.astar_jpt.State, goal: bayrob.core.astar_jpt.Goal, models: Dict, state_similarity: float = 0.2, goal_confidence: float = 0.2)

   Bases: :py:obj:`bayrob.core.astar_jpt.SubAStarBW`


   .. py:method:: stepcost(state, parent) -> float


   .. py:method:: h(state: bayrob.core.astar_jpt.State) -> float


   .. py:method:: plot_pos(path: List, title: str = None, conf: float = None, limx: Tuple = None, limy: Tuple = None, limz: Tuple = None, save: str = None, show: bool = True) -> plotly.graph_objs.Figure


   .. py:method:: plot_dir(path: List, title: str = None, conf: float = None, limx: Tuple = None, limy: Tuple = None, limz: Tuple = None, save: str = None, show: bool = True) -> plotly.graph_objs.Figure


   .. py:method:: plot_path(xvar, yvar, p: List, obstacles: List = None, title: str = None, save: str = None, show: bool = False, limx: Tuple = None, limy: Tuple = None) -> plotly.graph_objs.Figure


   .. py:method:: gendata(xvar, yvar, state, conf: float = None)


   .. py:method:: plot(node: jpt.trees.Node, **kwargs) -> plotly.graph_objs.Figure


