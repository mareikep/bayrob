bayrob.models.action
====================

.. py:module:: bayrob.models.action


Attributes
----------

.. autoapisummary::

   bayrob.models.action.datalogger


Classes
-------

.. autoapisummary::

   bayrob.models.action.Move
   bayrob.models.action.TrajectorySimulation


Module Contents
---------------

.. py:data:: datalogger

.. py:class:: Move(degu: float = 0.01, distu: float = 0.05)

   .. py:attribute:: DEG_U
      :value: 0.01



   .. py:attribute:: DIST_U
      :value: 0.05



   .. py:attribute:: STEPSIZE
      :value: 1



   .. py:method:: rotate(x: float, y: float, deg: float) -> (float, float)
      :staticmethod:



   .. py:method:: turnleft(agent) -> None
      :staticmethod:



   .. py:method:: turnright(agent) -> None
      :staticmethod:



   .. py:method:: turndeg(agent, deg=45) -> None
      :staticmethod:



   .. py:method:: moveforward(agent, dist=1) -> None
      :staticmethod:



   .. py:method:: movestep(agent) -> None
      :staticmethod:



   .. py:method:: sampletrajectory(agent, actions=None, p=None, steps=10) -> numpy.ndarray
      :staticmethod:



   .. py:method:: plot(jpt_: jpt.trees.JPT, qvarx: jpt.variables.Variable, qvary: jpt.variables.Variable, evidence: Dict[jpt.variables.Variable, Any] = None, title: str = None, conf: float = None, limx: Tuple = None, limy: Tuple = None, limz: Tuple = None, save: str = None, show: bool = False) -> None
      :staticmethod:


      Plots a heatmap representing the overall `coverage` of the jpt for the given variables, i.e. the joint
      probability of these variables: P(qvarx, qvary [| evidence ]). Helps to identify areas not well represented
      by the tree.

      :param jpt_: The (conditional) tree to plot the overall coverage for
      :param qvarx: The first of two joint variables to show the coverage for
      :param qvary: The second of two joint variable to show the coverage for
      :param evidence: The evidence for the conditional probability represented (if present)
      :param title: The plot title
      :param conf:  A confidence value. Values below this threshold are set to 0. (= equal color for lowest value in plot)
      :param limx: The limits for the x-variable; determined from pdf intervals of jpt priors if not given
      :param limy: The limits for the y-variable; determined from pdf intervals of jpt priors if not given
      :param limz: The limits for the z-variable; determined from pdf intervals of jpt priors if not given
      :param save: The location where the plot is saved (if given)
      :param show: Whether the plot is shown
      :return: None



.. py:class:: TrajectorySimulation(x=10, y=10, probx=None, proby=None)

   .. py:method:: dir(prob) -> int


   .. py:method:: step(posx, posy) -> List[float]


   .. py:method:: sample(n=1, s=10, initpos=None) -> pandas.DataFrame


