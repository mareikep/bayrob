from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Iterable, Dict, Union, List, Tuple

import dnutils
from dnutils import ifnone

import jpt
from bayrob.application.astar_jpt_app import SubAStarBW_, SubAStar_
from bayrob.core.astar import BiDirAStar
from bayrob.core.astar_jpt import Goal, State
from bayrob.utils.constants import bayroblogger, bayrobjsonlogger
from jpt.trees import JPT
from jpt.distributions import Gaussian, Numeric
from jpt.variables import VariableMap

logger = dnutils.getlogger(bayroblogger, level=dnutils.DEBUG)
jsonlogger = dnutils.getlogger(bayrobjsonlogger)


class Search:
    """The Search object encapsulating required information for a `BayRoB` plan refinement."""

    def __init__(self):
        """The required attributes for the plan refinement are

        Attributes:
            goal    the goal specification
            init    the initial state
        """
        self.goal = {}
        self.goal_tolerances = {}
        self.init = {}
        self.init_tolerances = {}
        self.bwd = True

    def __str__(self):
        return (f'<Search('
                f'init: {", ".join([f"{k}: {v} ({self.init_tolerances.get(k)})" for k, v in self.init.items()])}; '
                f'goal: {", ".join([f"{k}: {v} ({self.goal_tolerances.get(k)})" for k, v in self.goal.items()])}; '
                f'dir: {"backward" if self.bwd else "forward"}'
                f')>')


class Query:
    """ """

    def __init__(self):
        self.evidence = {}
        self.model = None
        self.modelname = None
        self.querystr = ''
        self.queryvars = []
        self.plottype = None
        self.plot_tree = False

    def __str__(self):
        return (f'<Query('
                f'model name: {self.modelname}; '
                f'evidence: {", ".join([f"{k}: {v}" for k, v in self.evidence.items()])}; '
                f'evidence string: {self.querystr}; '
                f'query variables: {", ".join([str(p) for p in self.queryvars])}; '
                f'plot type: {self.plottype}'
                f')>')


class BayRoB:
    """The BayRoB reasoning system."""

    def __init__(
            self,
            query: Union[Query, Search] = None,
            datapaths=None,
    ):
        """The BayRoB system allows to either query individual action models or perform a search/plan refinement on
        all models.
        """
        self._query = query
        self._datapaths = ifnone(datapaths, [])
        self._reloadmodels()
        self._models = {}
        self._datasets = {}
        self.omitmodels = []
        self.search_mode = 'reverse'
        self._result = self.Result(query_object=query)

    class Result:
        """The result of either query or plan refinement in `BayRoB`. The object contains information about the success
         of the operation and provides information about possible errors or additional messages relevant to interpret
         the outcome."""
        def __init__(
                self,
                query_object,
                success=False,
                error=None,
                message=None,
                result=None
        ):
            self._qo = query_object
            self._type = "AStar" if isinstance(query_object, Search) else "JPT-Query"
            self._success = success
            self._error = error
            self._message = message
            self._result = result

        def __str__(self):
            nl = "\n"
            post = ""
            if self._type == "JPT-Query":
                post = nl.join([f"{str(k)}:{nl}{str(v)}" for k, v in self.result[1].items()])
            return (f'<Result('
                    f'For: {str(self.query_object)}{nl}'
                    f'{"Search" if self._type == "AStar" else "Query"} {"succeeded!" if self.success else f"failed with error: {self.error}."}{nl}'
                    f'Message: {self.message}{nl}'
                    f'Result of {"search" if self._type == "AStar" else "query"}: {nl}'
                    f'{f"Found path:{nl}{str(self.result)}" if self._type == "AStar" else f"Conditional tree:{nl}{str(self.result[0])}{nl}{nl}Posteriors: {nl}{post}"}; '
                    f')>')

        @property
        def query_object(self) -> Union[Search, Query]:
            return self._qo

        @query_object.setter
        def query_object(self, qo) -> None:
            self._qo = qo

        @property
        def success(self) -> bool:
            """Returns True if the task was executed without errors, otherwise False."""
            return self._success

        @property
        def error(self) -> str:
            """Returns True if the task was executed without errors, otherwise False."""
            return self._error

        @property
        def message(self) -> str:
            """A message with additional information possibly relevant to interpret the result."""
            return self._message

        @property
        def result(self) -> Union[Tuple[jpt.trees.JPT, VariableMap], List[State]]:
            """The result of the task execution. Either a sequence of actions (plan refinement) or a tuple
            `(cond, post)` of a conditional tree and posterior distributions after reasoning over an individual model.
             """
            return self._result

        def clear(self):
            """Remove all information of the result except the passed query object. """
            self._success = False
            self._error = None
            self._message = None
            self._result = None

    def adddatapath(
            self,
            path
    ) -> None:
        """Adds a path to an action model containing the model file containing an :class:`jpt.trees.JPT` instance and
        a .parquet file containing the original data to train the model. After adding the path(s), the model is loaded.
        The model objects as well as the path to the datafiles are accessible using the attributes :attr:`models` and
        :attr:`datasets`.

        :param path:    The path(s) to an action model containing data and model files
        :type path:     str or iterable
        """
        if not isinstance(path, Iterable):
            path = [path]
        for p in path:
            if p not in self._datapaths:
                self._datapaths.append(p)
        logger.info(f'Loading models: {", ".join(self._datapaths)}')
        self._reloadmodels()
        self._reloaddatasets()

    def removedatapath(
            self,
            path
    ) -> None:
        """Removes one or more path(s) pointing to model and data files. The respective models are removed from the
        :attr:`models`.

        :param path:    The path(s) to an action model containing data and model files
        :type path:     str or iterable
        """
        if not isinstance(path, Iterable):
            path = [path]
        for p in path:
            if p in self._datapaths:
                self._datapaths.remove(path)
        self._reloadmodels()
        self._reloaddatasets()

    def _reloadmodels(self) -> None:
        self._models = dict(
            [
                (
                    Path(p).name,
                    JPT.load(str(treefile))
                )
                for p in self._datapaths
                for treefile in Path(p).rglob('*.tree') if treefile.name not in self.omitmodels
            ]
        )
        logger.info(f'Loaded models: {", ".join(list(self._models.keys()))}')

    def _reloaddatasets(self):
        self._datasets = dict(
            [
                (
                    Path(p).name,
                    treefile.parent.absolute().joinpath(f'{treefile.stem}.parquet')
                )
                for p in self._datapaths
                for treefile in Path(p).rglob('*.tree') if treefile.name not in self.omitmodels and treefile.parent.joinpath(f'{treefile.stem}.parquet').is_file()
            ]
        )
        logger.info(f'Datafiles: {", ".join(list(self._datasets.keys()))}')

    @property
    def query(self) -> Union[Query, Search]:
        """The :attr:`query` attribute is an instance of either :class:`Query` or :class:`Search` specifying the
        required information for reasoning over an individual model or refining a plan."""
        return self._query

    @query.setter
    def query(self, q) -> None:
        self._query = q

    @property
    def datapaths(self) -> list:
        """The datapaths containing the paths action models.

        :returns:   list of paths to the folders containing pickled :class:`jpt.trees.JPT` models and the .parquet data
                    file used for training the model
        :rtype:     List[str]
        """
        return self._datapaths

    @property
    def models(self) -> Dict[str, JPT]:
        """The action models used for reasoning.

        :returns: a mapping from model names to :class:`jpt.trees.JPT` models
        :rtype: Dict[str, JPT]
        """
        return self._models

    @property
    def datasets(self) -> Dict[str, str]:
        """The datasets containing the data used for training the models.

        :returns: a mapping from model names to the path of the .parquet data file used for training the model
        :rtype: Dict[str, str]
        """
        return self._datasets

    @property
    def result(self):
        """The result object to be processed by the calling entity (e.g. to visualize).

        :returns: the generated tree representation of the inference result
        :rtype: :class:`core.base.BayRoB.Result`
        """
        return self._result

    def search_astar(self) -> None:
        """This function performs a plan refinement by searching a path from the initial state to the goal state
        defined in the :class:`bayrob.core.base.Search` object passed to the class beforehand."""

        # initialize init state
        init = State({})
        for var, val in self.query.init.items():
            logger.info(f'search_astar got query variable {var} ({var.__class__.__name__}), {val}, tolerance: {self.query.init_tolerances.get(var)}')
            if var.domain is Numeric:
                dx = Gaussian(val, self.query.init_tolerances.get(var, 0.1)).sample(500)
                dist = Numeric()
                dist.fit(dx.reshape(-1, 1), col=0)
            else:
                distx_ = var.distribution()
                dist = distx_.set([(1 if x in val else 0) / len(val) for x in list(var.domain.values)])

            init[var.name] = dist

        # initialize goal state
        goal = Goal({})
        for var, val in self.query.goal.items():
            logger.info(f'search_astar got goal variable {var} ({var.__class__.__name__}), {val}, tolerance: {self.query.goal_tolerances.get(var)}')
            if var.domain is Numeric:
                dx = Gaussian(val, self.query.goal_tolerances.get(var, 0.1)).sample(500)
                dist = Numeric()
                dist.fit(dx.reshape(-1, 1), col=0)
            else:
                distx_ = var.distribution()
                dist = distx_.set([(1 if x in val else 0) / len(val) for x in list(var.domain.values)])

            goal[var.name] = dist

        # perform A* search
        if self.query.bwd:
            astar = SubAStarBW_
        # elif self.search_mode == 'bidir':
        #     astar = BiDirAStar
        else:
            astar = SubAStar_

        astar.verbose = dnutils.WARNING

        a_star = astar(
            init,
            goal,
            models={k: v for k, v in self.models.items() if k not in ['pr2', 'alarm']}
        )
        logger.info(f"Running {a_star.__class__.__name__} with {repr(init)} and {goal}", a_star.models.keys())
        try:
            path = list(a_star.search())

            if path:
                self._result._success = True
                self._result._msg = f"Path found!"
            else:
                self._result._success = False
                self._result._error = "No path found!"
                self._result._message = f'Please try another search.'

            if self.query.bwd:
                path.reverse()
        except:
            path = []
            self._result._success = False
            self._result._error = "Error"
            self._result._message = f'An error occurred during search: {traceback.format_exc()}'

        self._result._result = path

    def query_jpts(self) -> None:
        """This function performs a query to an action model as specified in the :class:`bayrob.core.base.Query`
        object passed to the class beforehand."""
        logger.info(f'Got query: {self._query}')

        if self._query.model is None:
            logger.error(f'Model is None', str(self._query))
            self._result._success = False
            self._result._error = f'Model is None'
            self._result._message = f'Please select valid model.'
            return

        # query single JPT trees according to query information
        cond = self._query.model.conditional_jpt(
            evidence=self._query.model.bind(
                {k: v for k, v in self._query.evidence.items() if k.name in self._query.model.varnames},
                allow_singular_values=False
            ),
            fail_on_unsatisfiability=False
        )

        self._result._success = cond is not None

        # if query is unsatisfiable, update result and error messages
        if cond is None:
            logger.error(f'Query is unsatisfiable', str(self._query))
            self._result._success = False
            self._result._error = f'Unsatisfiable query'
            self._result._message = f'Please try another query.'
            return

        # compute posteriors for plots in webapp
        post = cond.posterior(
            variables=[v for v in self._query.model.variables if v.name not in self._query.evidence],
        )

        logger.info(f"Nodes in original tree: {len(self._query.model.allnodes)}; nodes in conditional tree: {len(cond.allnodes)}")

        self._result._result = (cond, post)
