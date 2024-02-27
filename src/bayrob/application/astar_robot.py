from typing import List

import dnutils
from dnutils import edict, out

from bayrob.application.hypothesis_robot import Hypothesis_Robot, Hypothesis_Robot_FW, Hypothesis_Robot_BW
from bayrob.core.astar import AStar
from bayrob.core.hypothesis import Step
from bayrob.utils.constants import bayroblogger
from bayrob.utils.utils import tovariablemapping, satisfies
from jpt.variables import VariableMap

logger = dnutils.getlogger(bayroblogger, level=dnutils.DEBUG)


class FWAStar(AStar):
    """Forward-implementation of the A* algorithm in the context of the BayRoB system.
    """
    def __init__(self, start, goal, models=None, **kwargs):
        """
        :param start: a mapping from variable names to values
        :type start: dict
        :param goal: a variable-interval mapping
        :type goal: dict
        :param models: a mapping from tree names to trees
        :type models: dict
        """
        self.models = models
        goal_ = tovariablemapping(goal, models)
        # create initial 'empty' hypothesis
        start_ = Hypothesis_Robot_FW([], start, goal_, steps=None)
        start_.goal = goal_
        start_.execchain(goal_, models)

        super().__init__(start_, goal_, **kwargs)

    def generate_candidatesteps(self, node, query) -> List[Step]:
        """Generates potential next steps by restricting the trees to only contain leaves that are reachable from the
        current position.

        :param node: the bayrob hypothesis so far (current node)
        :type node: calo.application.hypothesis_robot.Hypothesis_Robot
        :param query: a variable-interval mapping
        :type query: jpt.variables.VariableMap
        """
        evidence = node.result or tovariablemapping(node.init, self.models)
        condtrees = [[tn, tree.conditional_jpt(evidence=evidence)] for tn, tree in self.models.items()]
        return [Step({idx: 1}, leaf, treename, query) for treename, tree in condtrees for idx, leaf in tree.leaves.items()]  # FIXME: confs!
        # return [Step(confs, path, treename, query) for treename, tree in self.models.items() for idx, (confs, path) in enumerate(tree.reverse({k.name: v for k, v in query.items()}, confidence=.15))]

    def generate_successors(self, node) -> List[Hypothesis_Robot]:
        ''' Builds one candidate hypothesis for each of the generated next steps by copying current hypothesis and
        adding the step.

        :param node: the bayrob hypothesis so far (current node)
        :type node: calo.application.hypothesis_robot.Hypothesis_Robot
        '''
        query = tovariablemapping(node.tempgoal, self.models)  # in forward a*, this is the current position
        successors = []
        # build new hypotheses (=nodes) from candidate steps
        for succ in self.generate_candidatesteps(node, query):

            node_ = node.copy(succ.idx)
            # append_step automatically updates tempgoal
            node_.append_step(succ)
            # precondition of new hypothesis is result of old hypothesis
            node_.precond = tovariablemapping(edict({k.name: v.result for k, v in node.result.items()}), self.models)
            node_.execchain(query, self.models)
            if tovariablemapping(node_.tempgoal, self.models) in [s.query for s in node_.steps]:
                logger.info(f'Dropping predecessor {succ.idx} for hypothesis {node_.id} as it would create loops.')
                continue

            successors.append(node_)
        return successors

    def isgoal(self, node, onlygoal=False) -> bool:
        # goal check needs to satisfy two conditions:
        # - the resulting position must somewhat match the intended goal position
        # - the precondition of the first step of the chain must match the init_pos position
        goalmatch = satisfies(tovariablemapping({v.name: res.result for v, res in node.result.items()}, self.models), self.target)

        # out('-----------------------------')
        # out(f'GOAL CHECK FOR NODE: {node.id}; ONLYGOAL: {onlygoal}')
        # out('-----------------------------')
        # out('RESULT', tovariablemapping({v.name: res.result for v, res in node.result.items()}, self.models))
        # out('TARGET', self.target)
        # out('satisfies goal', satisfies(tovariablemapping({v.name: res.result for v, res in node.result.items()}, self.models), self.target))
        # out('-----------------------------')
        # out('START', tovariablemapping(self.init_pos.init, self.models), type(self.init_pos.init))
        # out('PRECOND', node.precond)
        # out('satisfies precondition', satisfies(tovariablemapping(self.init_pos.init, self.models), node.precond))
        # out('-----------------------------')
        # out('satisfies overall isgoal', satisfies(tovariablemapping({v.name: res.result for v, res in node.result.items()}, self.models), self.target) and satisfies(tovariablemapping(self.init_pos.init, self.models), node.precond))
        # out('-----------------------------')

        return goalmatch

    def retrace_path(self, node) -> Hypothesis_Robot:
        return node


class BWAStar(AStar):
    """Backward-implementation of the A* algorithm in the context of the BayRoB system.
    """
    def __init__(self, start, goal, models=None, **kwargs):
        self.models = models
        goal_ = tovariablemapping(goal, models)
        start_ = Hypothesis_Robot_BW([], start, goal_, steps=None)
        start_.goal = goal_
        start_.execchain(goal_, models)

        super().__init__(start_, goal_, **kwargs)

    def generate_candidates(self, query) -> List[Step]:
        """

        :param query: a variable-interval mapping
        :type query: jpt.variables.VariableMap
        """
        return [Step(confs, path, treename, query) for treename, tree in self.models.items() for idx, (confs, path) in enumerate(tree.reverse({k.name: v for k, v in query.items()}, confidence=.15))]

    def generate_successors(self, node) -> List[Hypothesis_Robot]:
        query = tovariablemapping(node.tempgoal, self.models)
        successors = []
        for pred in self.generate_candidates(query):

            node_ = node.copy(pred.idx)
            # addstep automatically updates tempgoal
            node_.prepend_step(pred)
            # new precondition of hypothesis is old precondition plus path of newly added step
            node_.precond = tovariablemapping(edict({k.name: v for k, v in node.precond.items()}) + {v.name: val for v, val in pred.path.items()}, self.models)
            node_.execchain(query, self.models)
            if tovariablemapping(node_.tempgoal, self.models) in [s.query for s in node_.steps]:
                logger.info(f'Dropping predecessor {pred.idx} for hypothesis {node_.id} as it would create loops.')
                continue

            successors.append(node_)
        return successors

    def isgoal(self, node, onlygoal=False) -> bool:
        # goal check needs to satisfy two conditions:
        # - the resulting position must somewhat match the intended goal position
        # - the precondition of the first step of the chain must match the init_pos position
        goalmatch = satisfies(tovariablemapping({v.name: res.result for v, res in node.result.items()}, self.models), self.target)
        startmatch = satisfies(tovariablemapping(self.start.init, self.models), node.precond)

        # out('-----------------------------')
        # out(f'GOAL CHECK FOR NODE: {node.id}; ONLYGOAL: {onlygoal}')
        # out('-----------------------------')
        # out('RESULT', tovariablemapping({v.name: res.result for v, res in node.result.items()}, self.models))
        # out('TARGET', self.target)
        # out('satisfies goal', satisfies(tovariablemapping({v.name: res.result for v, res in node.result.items()}, self.models), self.target))
        # out('-----------------------------')
        # out('START', tovariablemapping(self.init_pos.init, self.models), type(self.init_pos.init))
        # out('PRECOND', node.precond)
        # out('satisfies precondition', satisfies(tovariablemapping(self.init_pos.init, self.models), node.precond))
        # out('-----------------------------')
        # out('satisfies overall isgoal', satisfies(tovariablemapping({v.name: res.result for v, res in node.result.items()}, self.models), self.target) and satisfies(tovariablemapping(self.init_pos.init, self.models), node.precond))
        # out('-----------------------------')
        return goalmatch and (onlygoal or startmatch)

    def retrace_path(self, node) -> Hypothesis_Robot:
        return node