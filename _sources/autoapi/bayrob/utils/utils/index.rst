bayrob.utils.utils
==================

.. py:module:: bayrob.utils.utils

.. autoapi-nested-parse::

   This module provides a number of wrapper classes and convenience functions.



Attributes
----------

.. autoapisummary::

   bayrob.utils.utils.logger
   bayrob.utils.utils.res1


Functions
---------

.. autoapisummary::

   bayrob.utils.utils.satisfies
   bayrob.utils.utils.tovariablemapping
   bayrob.utils.utils.res
   bayrob.utils.utils.generatemln
   bayrob.utils.utils.scatterplot
   bayrob.utils.utils.cov
   bayrob.utils.utils.pearson
   bayrob.utils.utils.pcc
   bayrob.utils.utils.pcc_
   bayrob.utils.utils.actions_to_treedata
   bayrob.utils.utils.dotproduct
   bayrob.utils.utils.length
   bayrob.utils.utils.angle
   bayrob.utils.utils.angledeg
   bayrob.utils.utils.vector
   bayrob.utils.utils.unit
   bayrob.utils.utils.distance
   bayrob.utils.utils.scale
   bayrob.utils.utils.add
   bayrob.utils.utils.pnt2line_
   bayrob.utils.utils.visualize_jpt_outer_limits
   bayrob.utils.utils.pnt2line
   bayrob.utils.utils.recent_example
   bayrob.utils.utils.fmt
   bayrob.utils.utils.dhms
   bayrob.utils.utils.euler_from_quaternion
   bayrob.utils.utils.discr_colors
   bayrob.utils.utils.uniform_numeric
   bayrob.utils.utils.urlable


Module Contents
---------------

.. py:data:: logger

.. py:function:: satisfies(sigma: jpt.variables.VariableMap, rho: jpt.variables.VariableMap) -> bool

   Checks if a state ``sigma`` satisfies the requirement profile ``rho``, i.e. ``φ |= σ``

   :param sigma: a state, e.g. a property-value mapping or position
   :param rho: a requirement profile, e.g. a property name-interval, property name-values mapping or position
   :returns: whether the state satisfies the requirement profile


.. py:function:: tovariablemapping(mapping, models) -> jpt.variables.VariableMap

.. py:function:: res(p: str) -> str

.. py:function:: generatemln(data, threshold=10)

   Expects a list of Example items and generates a template MLN from it as well as training DBs.



.. py:function:: scatterplot(*args, **kwargs)

.. py:function:: cov(x) -> numpy.ndarray

.. py:function:: pearson(data, use_tgts=True, use_fts=True, removenans=True, ignorenans=True, ignore=None) -> (List[List[str]], numpy.ndarray)

.. py:function:: pcc(X, fnames, ignorenans=True) -> (List[List[str]], numpy.ndarray)

   Gets a matrix of data, where each row is a sample, each column a variable and returns two matrices containing the
   feature names and the pearson correlation coefficients of the respective variables

   :param X:           the input data matrix, each row represents a data point, each column a variable
   :param fnames:      the names of the variables, i.e. len(fnames) = len(X[i])
   :param ignorenans:  whether to ignore nan values in the data or not. Caution! When ignoring NaN values, the covariance matrix is built from only the available values
   :return:            two matrices: the feature-feature name and values for the respective PCC


.. py:function:: pcc_(C, i, j) -> float

   Calculates the PCC (Pearson correlation coefficient) for the respective variables of the given indices

   :param C:   The covariance matrix for the variables
   :param i:   The index of the first variable to correlate
   :param j:   The index of the second variable to correlate
   :return:    (float) the pcc of the variables with the indices `i` and `j`


.. py:function:: actions_to_treedata(el, ad, idname='id') -> dict

   For tree visualization; assuming ad is pd.DataFrame, el pd.Series; from NEEM data csv files like:

       id                  type            startTime        endTime          duration       success  failure  parent          next          previous  object_acted_on  object_type  bodyPartsUsed  arm  grasp  effort
       Action_IRXOQHDJ     PhysicalTask    1600330068.38499 1600330375.44614 307.061154842377 True
       Action_YGLTFJUW     Transporting    1600330074.30271 1600330375.40287 301.100160360336 True          Action_IRXOQHDJ
       Action_RTGJLPIV     LookingFor      1600330074.64896 1600330074.79814 0.149180889129639 True          Action_YGLTFJUW
       Action_HNLQFJCG     Accessing       1600330075.04209 1600330075.14547 0.103375196456909 True          Action_YGLTFJUW



.. py:function:: dotproduct(v1, v2)

.. py:function:: length(v)

.. py:function:: angle(v1, v2)

.. py:function:: angledeg(v1, v2)

.. py:function:: vector(v1, v2)

.. py:function:: unit(v)

.. py:function:: distance(p0, p1)

.. py:function:: scale(v, sc)

.. py:function:: add(v1, v2)

.. py:function:: pnt2line_(pnt: Union[List, Tuple], start: Union[List, Tuple], end: Union[List, Tuple]) -> Tuple

   Given a line with coordinates 'start' and 'end' and the coordinates of a point 'pnt' the proc returns the shortest
   distance from pnt to the line and the coordinates of the nearest point on the line.

   Algorithm:

   1. Convert the line segment to a vector ('line_vec').
   2. Create a vector connecting start to pnt ('pnt_vec').
   3. Get the dot product of pnt_vec and line_vec ('dot').
   4. Find the squared length of the line vector ('line_len').
   5. If the line segment has length 0, terminate otherwise determine the projection distance from start/end.
   6. If t < 0, the nearest point would be on the extension closest to start, if t > 1, closest to end.
   7. Calculate the distance from pnt to the nearest point on the line segment ('dist').

   :param pnt: Union(List, Tuple)
   :param start:  Union(List, Tuple)
   :param end:  Union(List, Tuple)
   :return: Tuple


.. py:function:: visualize_jpt_outer_limits(models: Dict[str, jpt.trees.JPT]) -> None

.. py:function:: pnt2line(pnt: Union[List, Tuple], start: Union[List, Tuple], end: Union[List, Tuple]) -> Tuple[float, List]

   Given a line with coordinates 'start' and 'end' and the coordinates of a point 'pnt' the proc returns the shortest
   distance from pnt to the line and the coordinates of the nearest point on the line.

   :param pnt: The coordinates of the point or origin
   :param start: The coordinates of the start of the line
   :param end: The coordinates of the end of the line
   :return: Tuple of (distance, nearest_point_coordinates)


.. py:function:: recent_example(p: str = '.', pattern: str = None, pos=-1) -> str

   Return the name of the folder most recently created (assuming the folders are named in the given pattern, which is
   used for training robot action data)



.. py:function:: fmt(val, prec=2, positive=False)

.. py:function:: dhms(td)

.. py:function:: euler_from_quaternion(x, y, z, w, degree=True)

   Convert a quaternion into euler angles (roll, pitch, yaw)

   roll is rotation around x in radians (counterclockwise)
   pitch is rotation around y in radians (counterclockwise)
   yaw is rotation around z in radians (counterclockwise)


.. py:function:: discr_colors(size: int)

.. py:function:: uniform_numeric(a: float, b: float) -> jpt.distributions.Numeric

.. py:function:: urlable(text)

.. py:data:: res1

