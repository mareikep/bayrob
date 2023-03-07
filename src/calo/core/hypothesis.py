from __future__ import annotations

from dnutils import edict, ifnone

from calo.core.astar import Node
from jpt.variables import VariableMap


class Step:
    '''
    One Step of a :class:`calo.core.base.Hypothesis` is one reversed path through one tree, representing one action
    execution.
    '''
    def __init__(self, confs, path, tree, query):
        """
        :param confs: mapping from variables to a probability (confidence), that their expected value lies in the
        interval defined by the user
        :type confs: Dict[jpt.variables.Variable, float]
        :param path: a path from a leaf node to the root representing one action execution ('Step')
        :type path: List[jpt.tree.Node]
        :param tree: the tree this step occurs in
        :type tree: jpt.tree.JPT
        """
        self.confs = confs
        self.leaf = path[0] if path is not None else None
        self._path = self.leaf.path if self.leaf else None
        self.query = query
        self.tree = tree
        self.value = self.leaf.value if self.leaf else None

    def copy(self) -> Step:
        s_ = Step(self.confs, [self.leaf], self.tree, self.query)
        s_._path = VariableMap([(var, val) for var, val in self._path.items()])
        s_.value = dict(self.value)
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
        return f'{self.tree}-{self.leaf.idx}'

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


class Hypothesis(Node):
    """Semi-abstract implementation for Node class for A* algorithm in the context of
    the CALO system. Not to be instantiated directly.
    Inheriting classes have to implement step cost (g) and heuristic (h) functions.
    A Hypothesis is one possible trail through multiple (reversed) Trees, representing executing multiple actions
    subsequently to meet given criteria/requirements.
    """

    def __init__(self, idx, steps=None):
        self.identifiers = []
        self.id = idx
        self.steps = ifnone(steps, [])
        self._goal = None
        self._tempgoal = None
        self.result = VariableMap()
        self.precond = VariableMap()
        super().__init__()

    @property
    def goal(self) -> VariableMap:
        return self._goal

    @goal.setter
    def goal(self, goal) -> None:
        self._goal = goal
        self._tempgoal = goal

    @property
    def id(self) -> str:
        return 'H_{}'.format('.'.join([str(s.name) for s in self.steps]))

    @id.setter
    def id(self, idx) -> None:
        self.identifiers.extend(idx)

    def g(self) -> float:
        raise NotImplementedError

    def h(self) -> float:
        raise NotImplementedError

    @property
    def tempgoal(self) -> VariableMap:
        return self._tempgoal

    def _update_tempgoal(self) -> None:
        if not self.steps:
            self._tempgoal = self.goal
        else:

            self._tempgoal = edict({k.name: v for k, v in self._tempgoal.items()}) + {k.name.replace('_in', '_out'): v for k, v in self.steps[0].path.items()}

    def execchain(self, query, trees) -> None:
        # previous prediction
        prevpred = {}

        # probability that execution of chain produces desired output
        perf = 1.

        for sidx, step in enumerate(self.steps):
            # TODO: variables = all targets or only targets present in query?
            # exp = step.tree.expectation(variables=[t for t in step.tree.targets if t.name in query], evidence=step.path, fail_on_unsatisfiability=False)
            # leaf = step.tree.apply({k: k.domain.label2value(v) for k, v in step.path.items()})
            # exp = step.tree.expectation(variables=step.tree.targets, evidence=step.path, fail_on_unsatisfiability=False)
            if step.tree in trees:
                leaf = trees[step.tree].apply({k: k.domain.label2value(v) for k, v in step.path.items()})
                exp = trees[step.tree].expectation(variables=trees[step.tree].targets, evidence=step.path, fail_on_unsatisfiability=False)

                # FIXME for debugging. remove
                if exp is None:
                    leaf = list(leaf)
                    exp = trees[step.tree].expectation(variables=trees[step.tree].targets, evidence=step.path, fail_on_unsatisfiability=False)

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

    def addstep(self, candidate) -> None:
        self.steps.insert(0, candidate)
        self._update_tempgoal()

    def copy(self, idx) -> Hypothesis:
        if idx is None:
            hyp_ = self.__class__(self.identifiers)
        else:
            hyp_ = self.__class__([idx] + self.identifiers)
        hyp_.steps = [s.copy() for s in self.steps]
        hyp_.goal = {k: v for k, v in self.goal.items()}
        return hyp_

    def tojson(self) -> dict:
        return {'identifier': self.id, 'result': self.result, 'steps': [s.tojson() for s in self.steps]}

    def __lt__(self, other) -> bool:
        return self.f < other.f

    def __str__(self) -> str:
        return '{}: {}; {} steps>'.format(self.id, self.result, len(self.steps))

    def __repr__(self) -> str:
        return '<{} "{}" at 0x{}>'.format(self.__class__.__name__, self.id, hash(self))

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return hash((Hypothesis, tuple(self.steps)))

    def __add__(self, other):
        # TODO: create new merged hypothesis from two -> stitch paths together
        raise NotImplementedError

    def plot(self) -> None:
        raise NotImplementedError
