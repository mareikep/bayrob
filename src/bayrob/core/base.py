from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Iterable, Dict, Union

import dnutils
from dnutils import ifnone

from bayrob.application.astar_jpt_app import SubAStarBW_, SubAStar_
from bayrob.core.astar import BiDirAStar
from bayrob.core.astar_jpt import Goal, State
from bayrob.utils.constants import bayroblogger, bayrobjsonlogger
from jpt import JPT
from jpt.distributions import Gaussian, Numeric

logger = dnutils.getlogger(bayroblogger, level=dnutils.DEBUG)
jsonlogger = dnutils.getlogger(bayrobjsonlogger)


class Search:
    def __init__(self):
        self.goal = {}
        self.goal_tolerances = {}
        self.init = {}
        self.init_tolerances = {}

    def __str__(self):
        return (f'<Search('
                f'init: {", ".join([f"{k}: {v}" for k, v in self.init.items()])}; '
                f'goal: {", ".join([f"{k}: {v}" for k, v in self.goal.items()])}; '
                f')>')


class Query:
    def __init__(self):
        self.query = {}
        self.model = None
        self.modelname = None
        self.querystr = ''
        self.plotvars = []
        self.plottype = None

    def __str__(self):
        return (f'<Query('
                f'model name: {self.modelname}; '
                f'query: {", ".join([f"{k}: {v}" for k, v in self.query.items()])}; '
                f'query string: {self.querystr}; '
                f'plot vars: {", ".join([str(p) for p in self.plotvars])}; '
                f'plot type: {self.plottype}'
                f')>')


class Result:
    def __init__(
            self,
            query_object,
    ):
        self.query_object = query_object
        self._type = "AStar" if isinstance(query_object, Search) else "JPT-Query"
        self.success = False
        self.error = None
        self.message = None
        self.result = None

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

    def clear(self):
        self.success = False
        self.error = None
        self.message = None
        self.result = None


class BayRoB:
    """The BayRoB reasoning system."""

    def __init__(
            self,
            query: Union[Query, Search] = None,
            datapaths=None,
    ):
        """The requirement profile mapping property names to either :class:`jpt.base.intervals.ContinuousSet` or list
        of values (for symbolic variables).

        :Example:

            >>>  query = {
                            `Cohesive Energy`: ContinuousSet(min, max),
                            `Melting Temperature`: ...,
                            `Rauheit`: ...,
                            `Shear Modulus`: ...,
                            `Enthalpy`: ...
                            }

        """
        self._query = query
        self._datapaths = ifnone(datapaths, [])
        self.reloadmodels()
        self._models = {}
        self._datasets = {}
        self.omitmodels = []
        self.search_mode = 'reverse'
        self._result = None

    def adddatapath(
            self,
            path
    ) -> None:
        if not isinstance(path, Iterable):
            path = [path]
        for p in path:
            if p not in self._datapaths:
                self._datapaths.append(p)
        self.reloadmodels()
        self.reloaddatasets()

    def removedatapath(
            self,
            path
    ) -> None:
        if path in self._datapaths:
            self._datapaths.remove(path)
        self.reloadmodels()
        self.reloaddatasets()

    def reloadmodels(self) -> None:
        self._models = {}
        for p in self._datapaths:
            for treefile in Path(p).glob('*.tree'):
                logger.debug(f'Loading {treefile.name} from {p}...')
                if treefile.name in self.omitmodels: continue
                try:
                    self._models[treefile.name] = JPT.load(str(treefile))
                except Exception:
                    logger.error(f'Could not load tree {p} {treefile.name}:\n{traceback.print_exc()}')

    def reloaddatasets(self):
        self._datasets = {}
        for p in self._datapaths:
            for treefile in Path(p).glob('*.tree'):
                logger.debug(f'Loading {treefile.name} from {p}...')
                if treefile.name in self.omitmodels: continue
                if treefile.parent.joinpath(f'000-{treefile.stem}.parquet').is_file():
                    self._datasets[treefile.name] = treefile.parent.absolute().joinpath(f'000-{treefile.stem}.parquet')

    @property
    def query(self) -> Union[Query, Search]:
        """"""
        return self._query

    @query.setter
    def query(self, q) -> None:
        self._query = q

    @property
    def datapaths(self) -> list:
        """The datapaths containing the Regression trees.
        """
        return self._datapaths

    @property
    def models(self) -> Dict[str, JPT]:
        """The regression trees to limit the query to.

        :returns: the filenames of the pickled regression trees that are to be used for inference
        :rtype: list of str
        """
        return self._models

    @property
    def datasets(self) -> Dict[str, str]:
        """The paths of the datasets for the JPTs

        :returns: the filenames of the pickled regression trees that are to be used for inference
        :rtype: list of str
        """
        return self._datasets

    @models.setter
    def models(
            self,
            trees
    ) -> None:
        if all([isinstance(t, str) for t in trees]):
            self._models = dict([(t, JPT.load(os.path.join(p, t))) for p in self._datapaths for t in trees if t in os.listdir(p)])
        elif all([isinstance(t, JPT) for t in trees]):
            self._models = {'{}.tree'.format(t.name): t for t in trees}

    @property
    def result(self):
        """The result object to be processed by the calling entity (e.g. visualize)

        :returns: the generated tree representation of the inference result
        :rtype: core.base.Result
        """
        return self._result

    def search_astar(self) -> None:

        self._result = Result(self.query)

        # initialize init state
        init = State({})
        for var, val in self.query.init.items():
            logger.warning(f'Got {var} ({var.__class__.__name__}, {type(var)}), {val}')
            if var.domain is Numeric:
                dx = Gaussian(val, self.query.init_tolerances.get(val, 0.1)).sample(500)
                dist = Numeric()
                dist.fit(dx.reshape(-1, 1), col=0)
            else:
                distx_ = var.distribution()
                dist = distx_.set([(1 if x in val else 0) / len(val) for x in list(var.domain.values)])

            init[var.name] = dist

        # initialize goal state
        goal = Goal({})
        for var, val in self.query.goal.items():
            logger.warning(f'Got {var} ({var.__class__.__name__}, {type(var)}), {val}')
            if var.domain is Numeric:
                dx = Gaussian(val, self.query.goal_tolerances.get(val, 0.1)).sample(500)
                dist = Numeric()
                dist.fit(dx.reshape(-1, 1), col=0)
            else:
                distx_ = var.distribution()
                dist = distx_.set([(1 if x in val else 0) / len(val) for x in list(var.domain.values)])

            goal[var.name] = dist

        # perform A* search
        if self.search_mode == 'reverse':
            astar = SubAStarBW_
        elif self.search_mode == 'bidir':
            astar = BiDirAStar
        else:
            astar = SubAStar_

        astar.verbose = dnutils.WARNING

        a_star = astar(
            init,
            goal,
            models=self.models
        )
        path = list(a_star.search())

        if self.search_mode == 'reverse':
            path.reverse()

        if path:
            self._result.success = True
            self._result.msg = f"Path found!"
        else:
            self._result.success = False
            self._result.error = "No path found!"
            self._result.message = f'Please try another search.'

        self._result.result = path

    def query_jpts(self) -> None:

        self._result = Result(self._query)

        logger.info(f'Got query: {self._query}')

        if self._query.model is None:
            logger.error(f'Model is None', str(self._query))
            self._result.success = False
            self._result.error = f'Model is None'
            self._result.message = f'Please select valid model.'
            return

        # query single JPT trees according to query information
        cond = self._query.model.conditional_jpt(
            evidence=self._query.model.bind(
                {k: v for k, v in self._query.query.items() if k.name in self._query.model.varnames},
                allow_singular_values=False
            ),
            fail_on_unsatisfiability=False
        )

        self._result.success = cond is not None

        if cond is None:
            logger.error(f'Query is unsatisfiable', str(self._query))
            self._result.success = False
            self._result.error = f'Unsatisfiable query'
            self._result.message = f'Please try another query.'
            return

        post = cond.posterior(
            variables=[v for v in self._query.model.variables if v.name not in self._query.query],
        )

        logger.info(f"Nodes in original tree: {len(self._query.model.allnodes)}; nodes in conditional tree: {len(cond.allnodes)}")

        self._result.result = (cond, post)
