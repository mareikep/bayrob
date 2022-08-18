from __future__ import annotations

import os
import pprint

import dnutils
import numpy as np
from dnutils import edict, ifnone, out

from jpt import JPT
from jpt.base.intervals import ContinuousSet
from calo.utils.constants import calologger, calojsonlogger, projectnameUP
from calo.utils.utils import generatemln

logger = dnutils.getlogger(calologger)
jsonlogger = dnutils.getlogger(calojsonlogger)


class Step:
    '''
    One step of a hypothesis is one reversed path through one tree, representing one action execution
    '''
    def __init__(self, sim, steps, tree, treename=None):
        self.sim = sim
        self.leaf = steps[0] if steps is not None else None
        self.path = self.leaf.path if self.leaf else None
        self._treename = treename
        self.tree = tree
        self.value = self.leaf.value

    def copy(self) -> Step:
        s_ = Step(self.sim, self.leaf, None, treename=str(self._treename))
        s_.path = dict(self.path)
        s_.value = dict(self.value)
        return s_

    def tojson(self) -> dict:
        return {'name': self.name, 'sim': self.sim, 'params': {k: str(v) for k, v in self.leaf.path.items()}, 'samples': self.leaf.samples}

    @property
    def name(self):
        return f'{self._treename}-{self.leaf.idx}'

    @property
    def idx(self) -> str:
        return '{}:{}'.format(self.name if self.name is not None else '', self.leaf.idx)

    def __str__(self) -> str:
        return '<Step "{}" ({}), params: {}>'.format(self.name, self.sim, ', '.join(['{}= {}'.format(k, str(v)) for k, v in self.leaf.path.items()]))

    def __repr__(self) -> str:
        return '<{} name={} at 0x{}>'.format(self.__class__.__name__, self.name, hash(self))

    def __hash__(self) -> int:
        return hash((Step, self.name, tuple(self.path.items())))

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)


class Hypothesis:
    '''
    A Hypothesis is one possible trail through multiple (reversed) Trees, representing executing multiple actions
    subsequently to meet given criteria/requirements
    '''

    def __init__(self, idx, steps=None, queries=None):
        self.identifiers = []
        self.id = idx
        self._value = 1
        self._probability = 1
        self.steps = ifnone(steps, [])
        self.queries = ifnone(queries, [])
        self.result = {}

    @property
    def id(self) -> str:
        return 'H_{}'.format('.'.join([str(s.idx) for s in self.steps]))

    @id.setter
    def id(self, idx) -> None:
        self.identifiers.extend(idx)

    @property
    def value(self) -> float:
        return self._value

    @property
    def probability(self) -> float:
        return self._probability

    def execchain(self, trees, query) -> None:
        # previous prediction
        prevpred = {}

        # TODO what does value do? how well does chain match query?
        self._value = 1.

        # probability that execution of chain (so far) produces desired output
        self._probability = 1.

        for sidx, step in enumerate(self.steps):
            out('enter execchain loop', step.path, step.tree.targets)
            pred = step.tree.expectation(variables=step.tree.targets, evidence={var: ContinuousSet(var.domain.labels[val.lower], var.domain.labels[val.upper], val.left, val.right) for var, val in step.path.items()}, fail_on_unsatisfiability=False)
            out('result execchain', {var: val.format_result() for var, val in pred.items()}, step._treename)

            # result-change check: punish long chains with states that do not change the result
            if all([val in self.result and pred[val] == self.result[val] for val in list(set(pred).intersection(query))]):
                self._value = np.NINF
                self._probability = 0.

            # precondition check: if result of current step contains variable that is parameter in next step, the
            # values must match
            for pvar, pval in step.path.items():
                if pvar in prevpred:
                    if pval.contains(prevpred[pvar]):
                        self.result.update(pred)
                        self._value *= step.sim
                        self._probability *= step.sim
                    else:
                        # values do not match -> step cannot follow the previous step, therefore this chain is rendered useless
                        self._value = np.NINF
                        self._probability = 0.
                else:
                    self.result.update(pred)
                    logger.warning(self._value, self.probability, step.sim)
                    self._value *= step.sim
                    self._probability *= step.sim
            prevpred = pred

    def addstep(self, candidate) -> None:
        self.steps.insert(0, candidate)

    def copy(self, idx) -> Hypothesis:
        if idx is None:
            hyp_ = Hypothesis(self.identifiers)
        else:
            hyp_ = Hypothesis([idx] + self.identifiers)
        hyp_.steps = [s.copy() for s in self.steps]
        hyp_.queries = [dict(q) for q in self.queries]
        return hyp_

    def tojson(self) -> dict:
        return {'identifier': self.id, 'result': self.result, 'value': self.value, 'steps': [s.tojson() for s in self.steps]}

    def __str__(self) -> str:
        return '{}: ({:.2e}); {}; {} steps>'.format(self.id, self.value, self.result, len(self.steps))

    def __repr__(self) -> str:
        return '<{} "{}" ({:.2e}) at 0x{}>'.format(self.__class__.__name__, self.id, self.value, hash(self))

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

    def __init__(self):
        """The requirement profile mapping property names to :class:`jpt.base.intervals.ContinuousSet`.

        :Example:

            >>>  query = {
                            `Cohesive Energy`: ContinuousSet(min, max),
                            `Melting Temperature`: ...,
                            `Rauheit`: ...,
                            `Shear Modulus`: ...,
                            `Enthalpy`: ...
                            }

        """
        self.query = {}
        self.threshold = 0
        self._strategy = CALO.DFS
        self._datapaths = []
        self._models = {}
        self.usemodels = []
        self._hypotheses = []
        self._nodes = {}
        self._resulttree = None

    def adddatapath(self, path) -> None:
        if path not in self._datapaths:
            self._datapaths.append(path)

    def removedatapath(self, path) -> None:
        if path in self._datapaths:
            self._datapaths.remove(path)

    def reloadmodels(self) -> None:
        try:
            self._models = dict([(t, JPT.load(os.path.join(p, t))) for p in self._datapaths for t in os.listdir(p) if t.endswith('.tree') and (not self.usemodels or t in self.usemodels)])
        except Exception:
            logger.error(f'Could not load trees {[os.path.join(p, t) for p in self._datapaths for t in os.listdir(p) if t.endswith(".tree") and (not self.usemodels or t in self.usemodels)]}')

    @property
    def datapaths(self) -> list:
        """The datapaths containing the Regression trees.
        """
        return self._datapaths

    @property
    def models(self) -> dict[str, JPT]:
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
            self._strategy = CALO.DFS if strategy == 0 else CALO.BFS
        elif isinstance(strategy, str):
            self._strategy = CALO.DFS if strategy == 'DFS' else CALO.BFS
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
    def nodes(self) -> list:
        return self._nodes

    def infer(self) -> None:
        f"""Performs a query on {projectnameUP.__doc__} models using the :attr:`query`.
        """
        logger.info('Loading models...')

        self._resulttree = ResTree(self.query, threshold=self.threshold)
        self._hypotheses = []
        self.reloadmodels()
        logger.info(f'...done! Using models {", ".join(self.models)}')

        logger.info(f'Running hypothesis generation for query { {k: str(v) for k, v in self.query.items()} }, threshold {self.threshold}, strategy {"DFS" if self.strategy == 0 else "BFS"}')

        if self.strategy == CALO.DFS:
            self._generatepaths_dfs(self.query)
        else:
            self._generatepaths_bfs(self.query)
        logger.info(f'Found {len(self._hypotheses)} Hypothes{"is" if len(self._hypotheses) == 1 else "es"}\n{[str(h) for h in self._hypotheses]}')
        jsonlogger.info('Found Hypotheses', {k: str(v) for k, v in self.query.items()}, [h.tojson() for h in self._hypotheses])

    def _generatepaths_dfs(self, query) -> None:
        """Recursive, depth-first search variation of function to generate hypotheses
        (paths through multiple trees) in order to satisfy query.

        :param query: the requirement profile
        :type query: dict
        """
        self._generatepaths_dfs_rec(query, Hypothesis([]))

    def _generatepaths_dfs_rec(self, query, hyp) -> None:

        hyp.execchain(self.models, self.query)
        idx = len(hyp.steps)

        if hyp.probability < self._resulttree.threshold:
            logger.debug('{}Dropping'.format('  ' * idx), hyp.id, hyp.probability)
            return
        if self._satisfies(hyp.result, self.query):
            if self.prefixexists(hyp):
                indices = np.argwhere([self._prefixof(hyp, h) for h in self._hypotheses])
                isshorter = False
                for hidx in reversed(indices):
                    out('checking', hidx, hidx[0], self._hypotheses[hidx[0]])
                    if len(hyp.steps) < len(self._hypotheses[hidx[0]].steps):
                        isshorter = True
                        logger.debug('{}Removing hypothesis'.format('  ' * idx), self._hypotheses[hidx[0]].id)
                        self._hypotheses.remove(self._hypotheses[hidx[0]])
                if isshorter:
                    logger.debug('{}Adding shorter hypothesis'.format('  ' * idx), hyp.id)
                    self._hypotheses.append(hyp)
            else:
                logger.info('{}Found valid hypothesis. Adding'.format('  ' * idx), hyp.id)
                self._hypotheses.append(hyp)
            return
        candidates = self._generate_candidates(query)
        for cidx, c in enumerate(candidates):
            q = edict(query) + c.path - self.models['{}.tree'.format(c.leaf.treename)].targets
            if q in hyp.queries: continue
            h_ = hyp.copy(c.idx)
            h_.addstep(c)
            h_.queries.append(q)
            self._generatepaths_dfs_rec(q, h_)

    def _generatepaths_bfs(self, query) -> None:
        """Breadth-first search variation of function to generate hypotheses
         (paths through multiple trees) in order to satisfy `query`.
         The hypotheses are assembled from the last step to the first, prepending
         one step after another.

        :param query: the requirement profile
        :type query: dict
        """
        self._generatepaths_bfs_rec([query], [Hypothesis([])])

    def _generatepaths_bfs_rec(self, queries, hyps) -> None:
        if not hyps:
            return

        hyps_ = []
        queries_ = []
        for h, query in zip(hyps, queries):
            # run current hypothesis on trees
            h.execchain(self.models, self.query)
            idx = len(h.steps)

            # drop current hypothesis if its probability is below the threshold set by the user
            if h.probability < self._resulttree.threshold:
                logger.debug(f'{"  "*idx}Dropping hypothesis {h.id} with probability {h.probability} < {self._resulttree.threshold}')
                continue
            # add hypothesis to final result if it satisfies the given query
            if self._satisfies(h.result, self.query):
                # .. only if there is no equivalent hypothesis that is shorter (= has less steps)
                if not self.prefixexists(h):
                    logger.info(f'{"  "*idx}Found valid hypothesis. Adding {h.id}')
                    self._hypotheses.append(h)
                continue
            # generate candidates that match current query
            candidates = self._generate_candidates(query)
            logger.warning('CANDIDATES\n' + "\n".join([str(c) for c in candidates]))
            for cidx, c in enumerate(candidates):
                # update query such that it merges the old query with the requirements from the following steps
                logger.debug('BEFORE UPDATE', edict(query), c.name, c.path, c.tree.targets)
                logger.error('PATH LABELS', {var: ContinuousSet(var.domain.labels[val.lower], var.domain.labels[val.upper], val.left, val.right) for var, val in c.path.items()})

                cpath = {var: ContinuousSet(var.domain.labels[val.lower], var.domain.labels[val.upper], val.left, val.right) for var, val in c.path.items()}
                q = edict(query) + cpath - c.tree.targets

                logger.debug('AFTER UPDATE', q)

                if q in h.queries: continue
                h_ = h.copy(c.idx)
                h_.addstep(c)
                h_.queries.append(q)
                hyps_.append(h_)
                queries_.append(q)

        logger.debug('recursive call with')
        pprint.pprint(queries_)
        pprint.pprint(hyps_)
        self._generatepaths_bfs_rec(queries_, hyps_)

    def _generate_candidates(self, query) -> list[Step]:
        return [Step(sim, steps, tree, treename=treename) for treename, tree in self.models.items() for idx, (sim, steps) in enumerate(tree.reverse(query))]

    def prefixexists(self, hyp) -> bool:
        """Checks if any of the already added hypotheses is a prefix of ``hyp``, i.e. ``hyp`` contains unnecessary
        steps.

        :param hyp: the current hypothesis under consideration
        :type hyp: core.base.Hypothesis
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
        """Checks if a colored state ``sigma`` satisfies the requirement profile ``rho``, i.e. ``φ |= σ``

        :param sigma: a colored state, i.e. property-value mapping
        :param rho: a requirement profile, i.e. property-interval mapping
        :type sigma: dict
        :type rho: dict
        :returns: whether the state satisfies the requirement profile
        :rtype: bool
        """
        # if any property defined in original requirement profile cannot be found in result
        if any(x not in sigma for x in rho.keys()):
            return False

        # value of any resulting property needs to match interval defined in requirement profile (if present)
        for k, v in rho.items():
            if k in sigma:
                if not v.contains(sigma[k]):
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
            tree = JPT(name=name)
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
