from __future__ import annotations

import heapq
import itertools
import os
import pprint
import traceback
from collections import defaultdict
from typing import Dict, Any

import dnutils
import numpy as np
from dnutils import edict, ifnone, out, stop
from pathlib import Path

import jpt
from jpt import JPT
from jpt.base.intervals import ContinuousSet
from calo.utils.constants import calologger, calojsonlogger, projectnameUP
from calo.utils.utils import generatemln
from jpt.variables import VariableMap

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)
jsonlogger = dnutils.getlogger(calojsonlogger)


class Step:
    '''
    One Step of a :class:`calo.core.base.Hypothesis` is one reversed path through one tree, representing one action
    execution.
    '''
    def __init__(self, confs, path, tree, treename=None):
        """
        :param confs: mapping from variables to a probability (confidence), that their expected value lies in the
        interval defined by the user
        :type confs: Dict[jpt.variables.Variable, float]
        :param path: a path from a leaf node to the root representing one action execution ('Step')
        :type path: List[jpt.tree.Node]
        :param tree: the tree this step occurs in
        :type tree: jpt.tree.JPT
        :param treename: the name of the tree
        :type treename: str
        """
        self.confs = confs
        self.leaf = path[0] if path is not None else None
        self._path = self.leaf.path if self.leaf else None
        self.treename = treename
        self.tree = tree
        self.value = self.leaf.value if self.leaf else None

    def copy(self) -> Step:
        s_ = Step(self.confs, [self.leaf], None, treename=str(self.treename))
        s_._path = VariableMap([(var, val) for var, val in self._path.items()])
        s_.value = dict(self.value)
        s_.tree = self.tree
        return s_

    @property
    def path(self) -> VariableMap:
        '''Contains the label representation of the values'''
        return VariableMap([(var, var.domain.value2label(val)) for var, val in self._path.items()])

    @property
    def pathval(self) -> VariableMap:
        '''Contains the internal represenation of the values'''
        return self._path

    def tojson(self) -> dict:
        return {'name': self.name, 'confs': self.confs, 'params': {k: str(v) for k, v in self.leaf.path.items()}, 'samples': self.leaf.samples}

    @property
    def name(self):
        return f'{self.treename}-{self.leaf.idx}'

    @property
    def idx(self) -> str:
        return self.leaf.idx

    def __str__(self) -> str:
        return '<Step "{}" ({}), params: {}>'.format(self.name, self.confs, ', '.join(['{}= {}'.format(k, str(v)) for k, v in self.leaf.path.items()]))

    def __repr__(self) -> str:
        return '<{} name={} at 0x{}>'.format(self.__class__.__name__, self.name, hash(self))

    def __hash__(self) -> int:
        return hash((Step, self.name, tuple(self.path.items())))

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)


class Hypothesis:
    """
    A Hypothesis is one possible trail through multiple (reversed) Trees, representing executing multiple actions
    subsequently to meet given criteria/requirements.
    """

    def __init__(self, idx, steps=None, queries=None):
        self.identifiers = []
        self.id = idx
        self._performance = 0.
        self.steps = ifnone(steps, [])
        self.queries = ifnone(queries, [])
        self.result = VariableMap()
        self.precond = VariableMap()
        self.g = None
        self.h = None

    @property
    def id(self) -> str:
        return 'H_{}'.format('.'.join([str(s.idx) for s in self.steps]))

    @id.setter
    def id(self, idx) -> None:
        self.identifiers.extend(idx)

    @property
    def f(self) -> float:
        return self.g + self.h

    @property
    def performance(self) -> float:
        return self._performance

    def execchain(self, query) -> None:
        # previous prediction
        prevpred = {}

        # probability that execution of chain produces desired output
        perf = 1.

        for sidx, step in enumerate(self.steps):
            # TODO: variables = all targets or only targets present in query?
            # exp = step.tree.expectation(variables=[t for t in step.tree.targets if t.name in query], evidence=step.path, fail_on_unsatisfiability=False)
            leaf = step.tree.apply({k: k.domain.label2value(v) for k, v in step.path.items()})
            exp = step.tree.expectation(variables=step.tree.targets, evidence=step.path, fail_on_unsatisfiability=False)
            if exp is None:
                leaf = list(leaf)

            # TODO: make sure equality check still works!
            # result-change check: punish long chains with states that do not change the result
            if all([val in self.result and exp[val] == self.result[val] for val in [p for p in set(exp) if p.name in query]]):
                perf = 0.

            # precondition check: if result of current step contains variable that is parameter in next step, the
            # values must match
            # TODO: step.path or step.pathval? -> values or labels? SHOULD BE VALUES (step.pathval)
            for pvar, pval in step.pathval.items():
                if pvar in prevpred:
                    if pval.contains(prevpred[pvar]):
                        self.result.update(exp)
                        # FIXME: temporary solution for performance measure!
                        # self._performance *= step.confs
                        perf *= sum(step.confs.values())/len(step.confs)
                    else:
                        # values do not match -> step cannot follow the previous step, therefore this chain is
                        # rendered useless
                        perf = 0.
                else:
                    self.result.update(exp)
                    # FIXME: temporary solution for performance measure!
                    # self._performance *= step.confs
                    perf *= sum(step.confs.values())/len(step.confs)

            self._performance = perf
            prevpred = exp

    def addstep(self, candidate, query) -> None:
        self.steps.insert(0, candidate)
        self.queries.insert(0, query)

    def copy(self, idx) -> Hypothesis:
        if idx is None:
            hyp_ = Hypothesis(self.identifiers)
        else:
            hyp_ = Hypothesis([idx] + self.identifiers)
        hyp_.steps = [s.copy() for s in self.steps]
        hyp_.queries = [q.copy() for q in self.queries]
        return hyp_

    def loops(self) -> bool:
        '''A hypothesis contains loops if ad step occurs (more than) twice and the corresponding queries (i.e. states)
        match.
        '''
        for (s, q), (s_, q_) in itertools.combinations(zip(self.steps, self.queries), 2):
            if s.name == s_.name:
                if q == q_:
                    out(q == q_)
                    out(q)
                    out(q_)
                    return True
        return False

    def tojson(self) -> dict:
        return {'identifier': self.id, 'result': self.result, 'value': self.performance, 'steps': [s.tojson() for s in self.steps]}

    def __str__(self) -> str:
        return '{}: ({:.2e}); {}; {} steps>'.format(self.id, self.performance, self.result, len(self.steps))

    def __repr__(self) -> str:
        return '<{} "{}" ({:.2e}) at 0x{}>'.format(self.__class__.__name__, self.id, self.performance, hash(self))

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return hash((Hypothesis, tuple(self.steps)))


class ResTree:

    def __init__(self, query, threshold=0):
        self.threshold = threshold
        self.query = query
        self.root = None

    class Node:

        def __init__(self, parent, nodetext='', edgetext='', printnode=False):
            self.children = []
            self._parent = parent
            self.nodetext = nodetext
            self.edgetext = edgetext
            self.printnode = printnode
            self.isroot = False
            self.result = {}
            self.parameters = {}
            self.id = None
            self.sim = None

        def getroot(self) -> ResTree.Node:
            curnode = self
            while curnode.parent is not None:
                curnode = self.parent
            return curnode

        def append(self) -> None:
            if self.parent is not None:
                if self not in self.parent.children:
                    self.parent.children.append(self)
                self.parent.append()

        @property
        def nodetttext(self) -> str:
            if self.isroot:
                return ',<br>'.join(['{}: {}'.format(k, str(v)) for k, v in self.parameters.items()])
            else:
                return '{}<br><br>{}'.format(',<br>'.join(['{}: {}'.format(k, str(v)) for k, v in self.result.items()]), self.sim if self.printnode else '')

        @property
        def edgetttext(self) -> str:
            return '{}'.format(',<br>'.join(['{}: {}'.format(k, str(v)) for k, v in self.parameters.items()]))

        @property
        def parent(self) -> ResTree.Node:
            return self._parent

        @parent.setter
        def parent(self, parent) -> None:
            self._parent = parent

        def __repr__(self) -> str:
            return self.nodetext

        def __eq__(self, other) -> bool:
            return self.nodetext == other.nodetext


class CALO:
    """The CALO reasoning system."""

    DFS = 0
    BFS = 1
    ASTAR = 2

    def __init__(self, stepcost=None, heuristic=None):
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
        self._query = None
        self._state = None
        self.threshold = 0
        self._strategy = CALO.ASTAR
        self._datapaths = []
        self._models = defaultdict(JPT)
        self.omitmodels = []
        self._hypotheses = []
        self._nodes = {}
        self._resulttree = None
        self._stepcost = stepcost or self._stepcost_default
        self._heuristic = heuristic or self._heuristic_default

    def _stepcost_default(self, current) -> float:
        return len(current.steps) + 1

    def _heuristic_default(self, current, goal) -> float:
        '''
        :param current: alskd
        :type current:
        '''
        return 1.

    def adddatapath(self, path) -> None:
        if path not in self._datapaths:
            self._datapaths.append(path)
        self.reloadmodels()

    def removedatapath(self, path) -> None:
        if path in self._datapaths:
            self._datapaths.remove(path)
        self.reloadmodels()

    def reloadmodels(self) -> None:
        try:
            self._models = dict([(treefile.name, JPT.load(str(treefile))) for p in self._datapaths for treefile in Path(p).rglob('*.tree') if not treefile.name in self.omitmodels])
        except Exception:
            logger.error(f'Could not load trees {[str(treefile) for p in self._datapaths for treefile in Path(p).rglob("*.tree") if not treefile.name in self.omitmodels]}\n{traceback.print_exc()}')

    @property
    def query(self) -> jpt.variables.VariableMap:
        """"""
        return self._query

    @query.setter
    def query(self, q) -> None:
        self._query = self.tovariablemapping(q)

    @property
    def state(self) -> jpt.variables.VariableMap:
        """"""
        return self._state

    @state.setter
    def state(self, s) -> None:
        self._state = self.tovariablemapping(s)

    def tovariablemapping(self, mapping) -> jpt.variables.VariableMap:
        if isinstance(mapping, jpt.variables.VariableMap):
            return mapping
        elif all(isinstance(k, jpt.variables.Variable) for k, _ in mapping.items()):
            return VariableMap([(k, v) for k, v in mapping.items()])
        else:
            variables = [v for _, tree in self._models.items() for v in tree.variables]
            varnames = [v.name for v in variables]
            try:
                # there may be variables with identical names which are different python objects. It is assumed,
                # however, that they are semantically the same, so each of them has to be updated
                return VariableMap([(variables[i], v) for k, v in mapping.items() for i in [i for i, x in enumerate(varnames) if x == k]])
                # return VariableMap([(variables[varnames.index(k)], v) for k, v in mapping.items()])
            except ValueError:
                raise Exception(f'Variable(s) {", ".join([k for k in mapping.keys() if k not in varnames])} are not available in models. Available variables: {varnames}')

    @property
    def datapaths(self) -> list:
        """The datapaths containing the Regression trees.
        """
        return self._datapaths

    @property
    def models(self) -> defaultdict[str, JPT]:
        """The regression trees to limit the query to.

        :returns: the filenames of the pickled regression trees that are to be used for inference
        :rtype: list of str
        """
        return self._models

    @models.setter
    def models(self, trees) -> None:
        if all([isinstance(t, str) for t in trees]):
            self._models = dict([(t, JPT.load(os.path.join(p, t))) for p in self._datapaths for t in trees if t in os.listdir(p)])
        elif all([isinstance(t, JPT) for t in trees]):
            self._models = {'{}.tree'.format(t.name): t for t in trees}

    @property
    def strategy(self) -> int:
        """A search strategy to use for inference.

        :returns: 0 for depth-first search (DFS), 1 for breadth-first search (BFS)
        :rtype: int
        """
        return self._strategy

    @strategy.setter
    def strategy(self, strategy) -> None:
        if isinstance(strategy, int):
            self._strategy = {0: CALO.DFS, 1: CALO.BFS, 2: CALO.ASTAR}.get(strategy, None)
        elif isinstance(strategy, str):
            self._strategy = {'DFS': CALO.DFS, 'BFS': CALO.BFS, 'ASTAR': CALO.ASTAR, 'A*': CALO.ASTAR}.get(strategy, None)
        else:
            logger.error('Strategy is not a valid input', strategy)

    @property
    def resulttree(self) -> ResTree:
        """The resulting hypothesis as tree representation (e.g. for visualization purposes).

        :returns: the generated tree representation of the inference result
        :rtype: core.base.ResTree
        """
        return self._resulttree

    @property
    def hypotheses(self) -> list[Hypothesis]:
        """The resulting hypothesis from a forerun inference.

        :returns: the generated hypotheses
        :rtype: list of calo.core.base.Hypothesis
        """
        return self._hypotheses

    @property
    def nodes(self) -> dict:
        return self._nodes

    def infer(self) -> None:
        f"""Performs a query on {projectnameUP.__doc__} models using the :attr:`query`.
        """
        logger.info('Loading models...')

        self._resulttree = ResTree(self.query, threshold=self.threshold)
        self._hypotheses = []
        self.reloadmodels()
        logger.info(f'...done! Using models {", ".join(self.models.keys())}')

        logger.info(f'Running hypothesis generation for query { {k: str(v) for k, v in self.query.items()} }, threshold {self.threshold}, strategy {"DFS" if self.strategy == 0 else "BFS" if self.strategy == 1 else "A*"}')

        if self.strategy == CALO.DFS:
            self._generatepaths_dfs(self.query)
        elif self.strategy == CALO.BFS:
            self._generatepaths_bfs(self.query, self.state)
        elif self.strategy == CALO.ASTAR:
            logger.info('Calling A*...')
            self._generatepaths_astar(self.query)
            logger.info('after A*...')
        else:
            logger.error(f'{self.strategy} is not a valid strategy option.')

        logger.info(f'Found {len(self._hypotheses)} Hypothes{"is" if len(self._hypotheses) == 1 else "es"}\n{[str(h) for h in self._hypotheses]}')
        jsonlogger.info('Found Hypotheses', {k: str(v) for k, v in self.query.items()}, [h.tojson() for h in self._hypotheses])

    # A* Search Algorithm
    def _generatepaths_astar(self, query) -> None:
        # Initialize the open list
        open = []
        ocount = 0.

        # put the starting node on the open
        # list
        q0 = Hypothesis([])
        q0.g = 0.
        q0.h = 0.
        q0.execchain(query)

        heapq.heappush(open, (0, ocount, q0))
        ocount += 1.

        query_ = query

        # initialize the closed list
        closed = []
        ccount = 0.

        # while the open list is not empty
        while open:
            # find the node with the least f on the open list, call it "q" and pop it off the open list
            f, _, q = heapq.heappop(open)

            # GOAL CHECK: accept hypothesis if it satisfies the given query; i.e. if it is a goal state AND
            # preconditions are met
            if self._satisfies(self.tovariablemapping({v.name: res.result for v, res in q.result.items()}), query) and self._satisfies(q.precond, self.state):
                logger.warning(self.tovariablemapping({v.name: res.result for v, res in q.result.items()}))
                logger.error(self.state, self.tovariablemapping({v.name: val for v, val in q.steps[0].path.items()}))
                logger.info(f'Found valid hypothesis. Adding {q.id}')
                self._hypotheses.append(q)

            # generate q's predecessors
            predecessors = self._generate_candidates(query_)

            # for each predecessor
            for pred in predecessors:
                query_ = self.tovariablemapping(edict({k.name: v for k, v in query.items()}) + {k.name.replace('_in', '_out'): v for k, v in pred.path.items()})

                # if the current hypothesis already contains the query_, it will create loops by adding this predecessor
                # and will therefore be useless, so skip it
                if query_ in q.queries:
                    logger.debug(f'Dropping predecessor {pred.idx} for hypothesis {q_.id} as it would create loops.')
                    continue

                q_ = q.copy(pred.idx)
                q_.addstep(pred, query_)
                q_.precond = self.tovariablemapping({v.name: val for v, val in pred.path.items()})
                q_.execchain(query_)

                # compute both g and h for the new hypothesis containing this predecessor
                #   newhyp.g = distance between q_ and q
                #   newhyp.h = estimated distance from goal to q_; i.e. how close is current 'state' from the initial
                #   query
                q_.g = self._stepcost(q_)
                q_.h = self._heuristic(q_, query_)

                # if there is an equivalent hypothesis that is shorter (= has fewer steps), skip this successor
                if self.equivexists(q_):
                    logger.debug(f'Dropping hypothesis {q_.id}. Prefix or equivalent exists.')
                    continue

                if q_.loops():
                    logger.debug(f'Dropping hypothesis {q_.id}. Loops.')
                    continue

                # if a node with the same position as successor is in the CLOSED list which has a lower f than
                #   successor, skip this successor otherwise, add  the node to the open list
                # ... but only if it is not dropped due to a performance lower than required by the user
                if q_.performance < self._resulttree.threshold:
                    logger.debug(f'Dropping hypothesis {q_.id} with probability {q_.performance} < {self._resulttree.threshold}')
                    continue
                heapq.heappush(open, (q_.f, ocount, q_))
                ocount += 1.

            # push q on the closed list
            heapq.heappush(closed, (q.f, ccount, q))
            ccount += 1
            out('__________________________\n')

    # def _generatepaths_dfs(self, query) -> None:
    #     """Recursive, depth-first search variation of function to generate hypotheses
    #     (paths through multiple trees) in order to satisfy query.
    #
    #     :param query: the requirement profile
    #     :type query: jpt.variables.VariableMap
    #     """
    #     return self._generatepaths_dfs_rec(query, Hypothesis([]))
    #
    # def _generatepaths_dfs_rec(self, query, hyp) -> None:
    #     # TODO: UPDATE
    #     hyp.execchain(self.query)
    #     idx = len(hyp.steps)
    #
    #     if hyp.performance < self._resulttree.threshold:
    #         logger.debug('{}Dropping'.format('  ' * idx), hyp.id, hyp.performance)
    #         return
    #     if self._satisfies(hyp.result, self.query):
    #         if self.equivexists(hyp):
    #             indices = np.argwhere([self._prefixof(hyp, h) for h in self._hypotheses])
    #             isshorter = False
    #             for hidx in reversed(indices):
    #                 out('checking', hidx, hidx[0], self._hypotheses[hidx[0]])
    #                 if len(hyp.steps) < len(self._hypotheses[hidx[0]].steps):
    #                     isshorter = True
    #                     logger.debug('{}Removing hypothesis'.format('  ' * idx), self._hypotheses[hidx[0]].id)
    #                     self._hypotheses.remove(self._hypotheses[hidx[0]])
    #             if isshorter:
    #                 logger.debug('{}Adding shorter hypothesis'.format('  ' * idx), hyp.id)
    #                 self._hypotheses.append(hyp)
    #         else:
    #             logger.info('{}Found valid hypothesis. Adding'.format('  ' * idx), hyp.id)
    #             self._hypotheses.append(hyp)
    #         return
    #     candidates = self._generate_candidates(query)
    #     for cidx, c in enumerate(candidates):
    #         q = edict(query) + c.path - self.models['{}.tree'.format(c.leaf.treename)].targets
    #         if q in hyp.queries: continue
    #         h_ = hyp.copy(c.idx)
    #         h_.addstep(c)
    #         h_.queries.append(q)
    #         self._generatepaths_dfs_rec(q, h_)
    #
    # def _generatepaths_bfs(self, query, state) -> None:
    #     """Breadth-first search variation of function to generate hypotheses
    #      (paths through multiple trees) in order to satisfy `query`.
    #      The hypotheses are assembled from the last step to the first, prepending
    #      one step after another.
    #
    #     :param query: the requirement profile
    #     :type query: jpt.variables.VariableMap
    #     """
    #     self._generatepaths_bfs_rec([query], [state], [Hypothesis([])])
    #
    # def _generatepaths_bfs_rec(self, queries, states, hyps) -> None:
    #     if not hyps:
    #         return
    #
    #     hyps_ = []
    #     queries_ = []
    #     states_ = []
    #     for h, query in zip(hyps, queries):
    #         # run current hypothesis on trees
    #         h.execchain(self.query)
    #         idx = len(h.steps)
    #
    #         # drop current hypothesis if its probability is below the threshold set by the user
    #         if h.performance < self._resulttree.threshold:  # FIXME: find better performance measure for hypothesis (less dependent on length of chain)
    #             logger.debug(f'{"  "*idx}Dropping hypothesis {h.id} with probability {h.performance} < {self._resulttree.threshold}')
    #             continue
    #         # accept hypothesis if it satisfies the given query
    #         if self._satisfies(self.tovariablemapping({v.name.replace('_out', '_in'): res.result for v, res in h.result.items()}), self.query):
    #             # ... but only if there is no equivalent hypothesis that is shorter (= has fewer steps)
    #             if not self.equivexists(h):
    #                 logger.info(f'{"  "*idx}Found valid hypothesis. Adding {h.id}')
    #                 self._hypotheses.append(h)
    #             continue
    #         # generate candidates that match current query
    #         candidates = self._generate_candidates(query)
    #         for cidx, c in enumerate(candidates):
    #             # update query such that it merges the old query with the requirements from the following steps
    #             # q_ = edict(query) + c.path - [v.name for v in c.tree.targets]
    #             # q_ = self.tovariablemapping(edict({k.name: v for k, v in query.items()}) + {k.name.replace('_in', '_out'): v for k, v in c.path.items()})
    #             q_ = self.tovariablemapping(edict({k.name: v for k, v in query.items()}) + {k.name.replace('_in', '_out'): v for k, v in c.path.items()})
    #             # s_ = self.tovariablemapping(edict({k.name: v for k, v in state.items()}) + {v.name.replace('_out', '_in'): res.result for v, res in h.result.items()})
    #             # s_ = self.tovariablemapping(edict({k.name: v for k, v in state.items()}) + {k.name: v for k, v in c.path.items()})
    #
    #             if q_ in h.queries: continue
    #             h_ = h.copy(c.idx)
    #             h_.addstep(c)
    #             h_.queries.append(q_)
    #             hyps_.append(h_)
    #             # states_.append(s_)
    #             queries_.append(q_)
    #
    #     self._generatepaths_bfs_rec(queries_, states_, hyps_)

    def _generate_candidates(self, query) -> list[Step]:
        """

        :param query: a variable-interval mapping
        :type query: jpt.variables.VariableMap
        """
        return [Step(confs, path, tree, treename=treename) for treename, tree in self.models.items() for idx, (confs, path) in enumerate(tree.reverse({k.name: v for k, v in query.items()}))]

    def equivexists(self, hyp) -> bool:
        """Checks if any of the already added hypotheses is prefix of ``hyp``, i.e. ``hyp`` contains unnecessary
        steps, or TODO is an equivalent of ``hyp`` with a lower or equal performance

        :param hyp: the current hypothesis under consideration
        :type hyp: calo.core.base.Hypothesis
        :rtype: bool
        """

        return any([self._prefixof(hyp, hyp2) for hyp2 in self._hypotheses])

    def _prefixof(self, h1, h2) -> bool:
        """Returns True if Hypothesis `h1` is a prefix of `h2`, i.e. if the first n steps are identical for both
        hypotheses.
        """
        return all([s1 == s2 for s1, s2 in zip(h1.steps, h2.steps)])

    @staticmethod
    def _satisfies(sigma, rho) -> bool:
        """Checks if a state ``sigma`` satisfies the requirement profile ``rho``, i.e. ``φ |= σ``

        :param sigma: a state, e.g. a property-value mapping or position
        :type sigma: jpt.variables.VariableMap
        :param rho: a requirement profile, e.g. a property name-interval, property name-values mapping or position
        :type rho: jpt.variables.VariableMap
        :returns: whether the state satisfies the requirement profile
        :rtype: bool
        """
        # if any property defined in original requirement profile cannot be found in result
        if any(x not in sigma for x in rho.keys()):
            return False

        # value of any resulting variable needs to match interval defined in requirement (if present)
        for k, v in rho.items():
            if k in sigma:
                if isinstance(v, ContinuousSet):
                    if not v.contains_value(sigma[k]):
                        return False
                    # FIXME: check for expected value or enclosing interval?
                    # if not v.contains_interval(ContinuousSet(sigma[k].lower, sigma[k].upper)):
                    #     return False
                elif isinstance(v, list):
                    # case symbolic variable, v should be list
                    if not sigma[k] in v:
                        return False
                else:
                    if not sigma[k] == v:
                        return False
        return True

    @staticmethod
    def learn(name, data, model='tree', path='.') -> None:
        """Learns a new model with the given ``name`` using the training ``data``.

        :param name: the name for the model to be learned, e.g. the process name
        :param data: instances of calo.utils.utils.Example representing training data
        :param model: the type of model to learn, valid inputs are tree, mln
        :param path: the location to store the model in
        :type name: str
        :type data: list
        :type model: str
        :type path: str

        """
        if not data:
            logger.debug('Got no data. Nothing to do here.')
            return

        logger.debug('Training new model "{}" using {} examples...'.format(name, len(data)))

        logger.debug('Transforming training_data')
        for d in data:
            d.tosklearn()

        if 'tree' in model:
            logger.debug('Training regression tree')
            tree = JPT()
            tree.learn(data)
            tree.plot(filename='{}-tree'.format(name), directory=os.path.join(path, 'plots'))
            tree.pickle(os.path.join(path, '{}.tree'.format(name)))
        if 'mln' in model:
            logger.debug('Training MLN')
            mln, dbs = generatemln(data)

            mln.write()

            # TODO: learn MLN with dbs

            # # write databases to file
            # with open(os.path.join(locs.kb, 'dbs', 'databases_full_{}.db'.format(name)), 'w+') as fdb:
            #     for db in dbs:
            #         fdb.write('\n---\n')
            #         db.write(fdb, bars=False)
            #
            # # write mln to file
            # with open(os.path.join(locs.kb, 'mlns', 'predicates_{}.mln'.format(name)), 'w+') as fmln:
            #     mln.write(fmln)

        logger.debug('...done!')
