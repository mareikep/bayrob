bayrob.utils.plotlib
====================

.. py:module:: bayrob.utils.plotlib


Attributes
----------

.. autoapisummary::

   bayrob.utils.plotlib.logger
   bayrob.utils.plotlib.robot_positions


Functions
---------

.. autoapisummary::

   bayrob.utils.plotlib.defaultconfig
   bayrob.utils.plotlib.fig_from_json_file
   bayrob.utils.plotlib.fig_to_file
   bayrob.utils.plotlib.hextorgb
   bayrob.utils.plotlib.to_rgb
   bayrob.utils.plotlib.plot_pos
   bayrob.utils.plotlib.plot_dir
   bayrob.utils.plotlib.gendata_mixture
   bayrob.utils.plotlib.gendata
   bayrob.utils.plotlib.gendata_multiple
   bayrob.utils.plotlib.plotly_animation
   bayrob.utils.plotlib.plot_dists_layered
   bayrob.utils.plotlib.plot_heatmap
   bayrob.utils.plotlib.plot_tree_dist
   bayrob.utils.plotlib.pathdata
   bayrob.utils.plotlib.plot_path
   bayrob.utils.plotlib.plotly_pt
   bayrob.utils.plotlib.plotly_sq
   bayrob.utils.plotlib.plot_pt_sq
   bayrob.utils.plotlib.plot_scatter_quiver
   bayrob.utils.plotlib.build_constraints
   bayrob.utils.plotlib.filter_dataframe
   bayrob.utils.plotlib.plot_data_subset
   bayrob.utils.plotlib.gaussian
   bayrob.utils.plotlib.plot_multiple_dists
   bayrob.utils.plotlib.plot_tree_leaves
   bayrob.utils.plotlib.plot_typst_tree_json
   bayrob.utils.plotlib.plot_typst_jpt


Module Contents
---------------

.. py:data:: logger

.. py:function:: defaultconfig(fname=None, format='svg')

.. py:function:: fig_from_json_file(fname) -> plotly.graph_objs.Figure

.. py:function:: fig_to_file(fig, fname, configname=None, ftypes=None, forcecreate=False) -> None

   Writes figure to file with name `fname`. If multiple extensions are given in ftypes, the same figure is saved
   to multiple file formats.

   :param fig: The plotly Figure to save
   :param fname: The file name (including path) to save the figure to
   :param configname: The name to use for the plotly config when showing the figure
   :param ftypes:  A list of file types to save the figure as (e.g. ['.html', '.png']). If None, the file type is inferred from the suffix of `fname`.
   :param forcecreate: Whether to force creation of parent directories if they do not exist



.. py:function:: hextorgb(col)

.. py:function:: to_rgb(color, opacity=0.6)

.. py:function:: plot_pos(path: List, title: str = None, conf: float = None, limx: Tuple = None, limy: Tuple = None, limz: Tuple = None, d: List = None, save: str = None, show: bool = True, fun: str = 'heatmap') -> plotly.graph_objs.Figure

   Plot Heatmap representing distribution of `x_in`, `y_in` variables from a `path`.
   `path` is a list of (state, params) tuples generated in `test_astar_jpt_robotaction.test_astar_cram_path`.

   :param path: List of (state, params) tuples
   :param title: Plot title
   :param conf: Confidence threshold
   :param limx: X-axis limits
   :param limy: Y-axis limits
   :param limz: Z-axis limits
   :param save: Save path
   :param show: Whether to show plot
   :param fun: Plot type ('heatmap' or 'surface')



.. py:function:: plot_dir(path: List, title: str = None, conf: float = None, limx: Tuple = None, limy: Tuple = None, d: List = None, save: str = None, show: bool = True, fun: str = 'heatmap') -> plotly.graph_objs.Figure

.. py:function:: gendata_mixture(xvar, yvar, states, limx: Tuple = None, limy: Tuple = None, priors: List = None, numpoints=200)

.. py:function:: gendata(xvar, yvar, state, params: Dict = {}, conf: float = None)

   Generates data points

   :param xvar: X variable name
   :param yvar: Y variable name
   :param state: State dictionary
   :param params: Additional parameters
   :param conf: Confidence threshold



.. py:function:: gendata_multiple(vars: List[Tuple], states: List, params: Dict = {}, conf: float = None)

   Generates data points

   :param vars: List of (xvar, yvar) tuples
   :param states: List of states
   :param params: Additional parameters
   :param conf: Confidence threshold



.. py:function:: plotly_animation(data: List[Any], names: List[str] = None, title: str = None, save: str = None, show: bool = True, showbuttons: bool = True, speed: int = 100) -> plotly.graph_objs.Figure

   Animated Plot

   :param names:
   :param data:
   :param title:
   :param save:
   :param show:
   :return:



.. py:function:: plot_dists_layered(xvar: str, yvar: str, data: pandas.DataFrame, limx: Tuple = None, limy: Tuple = None, save: str = None, show: bool = False) -> plotly.graph_objs.Figure

.. py:function:: plot_heatmap(xvar: str, yvar: str, data: pandas.DataFrame, title: str = None, limx: Tuple = None, limy: Tuple = None, limz: Tuple = None, save: str = None, show: bool = True, text: str = None, fun: str = 'heatmap', showbuttons: bool = True, dark: bool = False) -> plotly.graph_objs.Figure

   Plot heatmap (animation) or 3D surface plot with plotly.

   :param xvar: The name of the x-axis of the heatmap and of the respective column in the `data` Dataframe
   :type xvar: str
   :param yvar: The name of the y-axis of the heatmap and of the respective column in the `data` Dataframe
   :type yvar: str
   :param data: Each row (!) consists of an entire dataset, such that multiple rows result in an animation, i.e. for an n x m heatmap, each `xvar` cell contains an array of shape (n,), each `yvar` cell contains an array of shape (m,) and the `z` cells contain arrays shaped (m,n). May also contain an optional column called `lbl` of shape (n,) or (m,n) which serves as custom information when hovering over data points.
   :type data: pd.DataFrame
   :param title: The title of the plot
   :type title: str
   :param limx: The limits of the x-axis. Determined automatically from data if not given
   :type limx: Tuple
   :param limy: The limits of the y-axis. Determined automatically from data if not given
   :type limy: Tuple
   :param save: a full path (including file name) to save the plot to.
   :type save: str
   :param show: whether to automatically open the plot in the default browser.
   :type show: bool



.. py:function:: plot_tree_dist(tree: jpt.trees.JPT, qvars: dict = None, qvarx: Any = None, qvary: Any = None, title: str = None, conf: float = None, limx: Tuple = None, limy: Tuple = None, limz: Tuple = None, save: str = None, show: bool = True) -> plotly.graph_objs.Figure

   Plots a heatmap representing the belief state for the agents' position, i.e. the joint
   probability of the x and y variables: P(x, y).

   :param title: The plot title
   :param conf:  A confidence value. Values below this threshold are set to 0. (= equal color for lowest value in plot)
   :param limx: The limits for the x-variable; determined from boundaries if not given
   :param limy: The limits for the y-variable; determined from boundaries if not given
   :param limz: The limits for the z-variable; determined from data if not given
   :param save: The location where the plot is saved (if given)
   :param show: Whether the plot is shown
   :return: None



.. py:function:: pathdata(xvar, yvar, p: List, exp: bool = False) -> List

.. py:function:: plot_path(xvar, yvar, p: List, d: List = None, obstacles: List = None, title: str = None, save: str = None, show: bool = False, dark: bool = False) -> plotly.graph_objs.Figure

.. py:function:: plotly_pt(pt: Tuple, dir: Tuple = None, name: str = None, color: str = 'rgb(15,21,110)') -> Any

.. py:function:: plotly_sq(area: Tuple, lbl: str = 'Goal', color: str = 'rgb(59,41,106)', legend: bool = True) -> Any

.. py:function:: plot_pt_sq(pt: Tuple, area: Tuple) -> plotly.graph_objs.Figure

.. py:function:: plot_scatter_quiver(xvar, yvar, data: pandas.DataFrame, title: str = None, save: str = None, show: bool = False) -> plotly.graph_objs.Figure

   Plot heatmap or 3D surface plot with plotly

   :param xvar: The name of the x-axis of the heatmap and of the respective column in the `data` Dataframe
   :param yvar: The name of the y-axis of the heatmap and of the respective column in the `data` Dataframe
   :param data: A Dataframe containing columns `xvar`, `yvar`, `
   :param title: The title of the plot
   :param save: a full path (including file name) to save the plot to.
   :param show: whether to automatically open the plot in the default browser.
   :return: Figure



.. py:function:: build_constraints(constraints)

.. py:function:: filter_dataframe(df: pandas.DataFrame, constraints) -> pandas.DataFrame

.. py:function:: plot_data_subset(df, xvar, yvar, constraints, limx=None, limy=None, save=None, show=False, plot_type='scatter', normalize=False, color='rgb(15,21,110)')

.. py:function:: gaussian(mean1, cov1, mean2, cov2, i=1)

.. py:function:: plot_multiple_dists(gaussians, limx: Tuple = None, limy: Tuple = None, save: str = None, show: bool = False)

.. py:function:: plot_tree_leaves(jpt: jpt.trees.JPT, varx: Any, vary: Any, limx: Tuple, limy: Tuple, save: str = None, color: str = None, show: bool = False) -> plotly.graph_objs.Figure

.. py:function:: plot_typst_tree_json(tree_data: dict, title: str = 'unnamed', filename: str or None = None, directory: str = None) -> str

   Generates an SVG representation of the generated regression tree.

   :param title: title of the plot
   :param filename: the name of the JPT (will also be used as filename; extension will be added automatically)
   :param directory: the location to save the SVG file to
   :param queryvars: the variables to be plotted in the graph
   :param view: whether the generated SVG file will be opened automatically
   :param max_symb_values: limit the maximum number of symbolic values that are plotted to this number
   :param nodefill: the color of the inner nodes in the plot; accepted formats: RGB, RGBA, HSV, HSVA or color name
   :param leaffill: the color of the leaf nodes in the plot; accepted formats: RGB, RGBA, HSV, HSVA or color name
   :param alphabet: whether to plot symbolic variables in alphabetic order, if False, they are sorted by probability (descending); default is False

   :return:   (str) the path under which the renderd image has been saved.



.. py:function:: plot_typst_jpt(jpt, title: str = 'unnamed', filename: str or None = None, directory: str = None, plotvars: Iterable[Any] = None, max_symb_values: int = 10, imgtype='svg', alphabet=False, svg=False) -> str

   Generates an SVG representation of the generated regression tree.

   :param title: title of the plot
   :param filename: the name of the JPT (will also be used as filename; extension will be added automatically)
   :param directory: the location to save the SVG file to
   :param plotvars: the variables to be plotted in the graph
   :param view: whether the generated SVG file will be opened automatically
   :param max_symb_values: limit the maximum number of symbolic values that are plotted to this number
   :param nodefill: the color of the inner nodes in the plot; accepted formats: RGB, RGBA, HSV, HSVA or color name
   :param leaffill: the color of the leaf nodes in the plot; accepted formats: RGB, RGBA, HSV, HSVA or color name
   :param alphabet: whether to plot symbolic variables in alphabetic order, if False, they are sorted by probability (descending); default is False

   :return:   (str) the path under which the renderd image has been saved.



.. py:data:: robot_positions

