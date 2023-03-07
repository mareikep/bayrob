from typing import List

import dnutils
from dnutils import edict, out

from calo.core.astar import AStar
from calo.core.hypothesis import Step
from calo.utils.constants import calologger
from calo.utils.utils import tovariablemapping, satisfies

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


class CALOAStar(AStar):
    """Semi-abstract implementation for A* algorithm in CALO system.
    Inheriting classes have to implement XXX"""
    def __init__(self, node, start, goal, models=None, **kwargs):
        self.models = models
        goal_ = tovariablemapping(goal, models)
        start_ = node([], start, goal_, steps=None)
        start_.goal = goal_
        start_.execchain(goal_, models)

        super().__init__(start_, goal_, **kwargs)

    def generate_candidates(self, query) -> List[Step]:
        """

        :param query: a variable-interval mapping
        :type query: jpt.variables.VariableMap
        """
        return [Step(confs, path, treename, query) for treename, tree in self.models.items() for idx, (confs, path) in enumerate(tree.reverse({k.name: v for k, v in query.items()}, confidence=.1))]

    def generate_successors(self, node):
        query = tovariablemapping(node.tempgoal, self.models)
        successors = []
        for pred in self.generate_candidates(query):

            node_ = node.copy(pred.idx)
            # addstep automatically updates tempgoal
            node_.addstep(pred)
            # new precondition of hypothesis is old precondition plus path of newly added step
            node_.precond = tovariablemapping(edict({k.name: v for k, v in node.precond.items()}) + {v.name: val for v, val in pred.path.items()}, self.models)
            node_.execchain(query, self.models)
            if tovariablemapping(node_.tempgoal, self.models) in [s.query for s in node_.steps]:
                logger.info(f'Dropping predecessor {pred.idx} for hypothesis {node_.id} as it would create loops.')
                continue

            successors.append(node_)
        return successors

    def isgoal(self, node, onlygoal=False):
        out('-----------------------------')
        out('GOAL CHECK FOR NODE:', node.id, 'ONLYGOAL:', onlygoal)
        out('-----------------------------')
        out('RESULT', tovariablemapping({v.name: res.result for v, res in node.result.items()}, self.models))
        out('TARGET', self.target)
        out('satisfies goal', satisfies(tovariablemapping({v.name: res.result for v, res in node.result.items()}, self.models), self.target))
        out('-----------------------------')
        out('START', tovariablemapping(self.start.init, self.models), type(self.start.init))
        out('PRECOND', node.precond)
        out('satisfies precondition', satisfies(tovariablemapping(self.start.init, self.models), node.precond))
        out('-----------------------------')
        out('satisfies overall isgoal', satisfies(tovariablemapping({v.name: res.result for v, res in node.result.items()}, self.models), self.target) and (onlygoal or satisfies(tovariablemapping(self.start.init, self.models), node.precond)))
        out('-----------------------------')
        return satisfies(tovariablemapping({v.name: res.result for v, res in node.result.items()}, self.models), self.target) and (onlygoal or satisfies(tovariablemapping(self.start.init, self.models), node.precond))

    def retrace_path(self, node):
        return node
