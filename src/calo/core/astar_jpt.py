import heapq
import operator
from collections import defaultdict
from functools import reduce
from typing import List, Dict, Any, Tuple

import dnutils
import pyximport
from dnutils import first
from jpt.distributions.quantile.quantiles import QuantileDistribution

import jpt
from calo.core.astar import AStar, Node
from calo.utils.constants import calologger
from jpt.distributions import Numeric
from jpt.variables import Variable

pyximport.install()
from jpt.base.intervals import R

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


class Goal(dict):

    # def __str__(self) -> str:
    #     return f'Goal< ({", ".join([f"{var}: {str(self[var])}" for var in self.keys()])}) >'
    def __str__(self) -> str:
        return f'Goal< ({", ".join([f"{var}: {str(self[var])}" for var in self.keys() if var not in ["tree", "leaf"]])}) ' \
               f'[{self.get("tree", "")}({self.get("leaf", "")})] >'

    def __repr__(self) -> str:
        return str(self)


class State(dict):

    def similarity(
            self,
            other: 'State'
    ) -> float:
        if set(self.keys()) != set(other.keys()):
            raise ValueError('Variable sets do not match.')
        return min(
            [
                type(self[var]).jaccard_similarity(var[var], other[var]) for var in self.items()
            ]
        )

    def __str__(self) -> str:
        print([type(t) for t in self.keys()], [k for k in self.keys()])
        return f'State< ({", ".join([f"{var}: {str(first(self[var].mpe()[1]))}" for var in self.keys()])}) >'

    def __repr__(self):
        return str(self)


class SubAStar(AStar):

    def __init__(
            self,
            initstate: State,
            goalstate: Goal,
            models: Dict,
            state_similarity: float = .9,
            goal_confidence: float = 1
    ):
        self.models = models
        super().__init__(
            initstate,
            goalstate,
            state_similarity=state_similarity,
            goal_confidence=goal_confidence
        )

    def init(self):
        init = Node(state=self.initstate, g=0., h=self.h(self.initstate), parent=None)
        heapq.heappush(self.open, (init.f, init))

    def generate_steps(
            self,
            node
    ) -> List[Any]:
        """Generates potential next steps by restricting the trees to only contain leaves that are reachable from the
        current position.

        :param node: the current node
        :type node: SubNode
        """

        evidence = {
            var: node.state[var].mpe()[1] for var in node.state.keys()
        }
        # evidence = {
        #     'x_in': node.state.posx.mpe()[1],
        #     'y_in': node.state.posy.mpe()[1],
        #     'xdir_in': node.state.dirx.mpe()[1],
        #     'ydir_in': node.state.diry.mpe()[1]
        # }

        condtrees = [
            [
                tn,
                tree.conditional_jpt(
                    evidence=tree.bind(
                        {k: v for k, v in evidence.items() if k in tree.varnames},
                        allow_singular_values=False
                    ),
                    fail_on_unsatisfiability=False)
            ] for tn, tree in self.models.items()
        ]

        return [(leaf, treename, tree) for treename, tree in condtrees if tree is not None for _, leaf in tree.leaves.items()]

    def generate_successors(
            self,
            node
    ) -> List[Node]:
        successors = []
        for succ, tn, t in self.generate_steps(node):

            # initialize new belief state for potential successor
            state = State(
                tree=tn,
                leaf=succ.idx,
            )

            for d in succ.distributions:
                # generate new distribution by shifting position delta distributions by expectation of position
                # belief state
                state[d] = Numeric().set(QuantileDistribution.from_cdf(succ[d].cdf.xshift(-node.state[d].expectation())))

            # if 'x_out' in succ.distributions:
            #     posx = Numeric().set(QuantileDistribution.from_cdf(succ.value['x_out'].cdf.xshift(-posx.expectation())))
            #
            # if 'y_out' in succ.distributions:
            #     posy = Numeric().set(QuantileDistribution.from_cdf(succ.value['y_out'].cdf.xshift(-posy.expectation())))
            #
            # # generate new orientation distribution by shifting orientation delta distributions by expectation of
            # # orientation belief state
            # if 'xdir_out' in succ.distributions:
            #     dirx = Numeric().set(QuantileDistribution.from_cdf(succ.value['xdir_out'].cdf.xshift(-dirx.expectation())))
            #
            # if 'ydir_out' in succ.distributions:
            #     diry = Numeric().set(QuantileDistribution.from_cdf(succ.value['ydir_out'].cdf.xshift(-diry.expectation())))

            successors.append(
                Node(
                    state=state,
                    g=node.g + self.stepcost(state),
                    h=self.h(state),
                    parent=node
                )
            )
        return successors

    def isgoal(
            self,
            node: Node
    ) -> bool:
        # true, if current belief state is sufficiently similar to goal
        # return node.state.posx.p(self.goal.posx) * node.state.posy.p(self.goal.posy) >= self.goal_confidence
        return reduce(operator.mul, [node.state[var].p(self.goal[var]) for var in self.goal]) >= self.goal_confidence


class SubAStarBW(SubAStar):

    def __init__(
            self,
            initstate: State,  # would be the goal state of forward-search
            goalstate: Goal,  # init state in forward-search
            models: Dict,
            state_similarity: float = .9,
            goal_confidence: float = 1
    ):

        super().__init__(
            initstate,
            goalstate,
            models=models,
            state_similarity=state_similarity,
            goal_confidence=goal_confidence
        )

    def init(self):
        # generate all the leaves that match the goal specification and push to open all nodes representing
        # 'predecessor' states from their preconditions
        n_ = Node(
            state=self.goal,
            g=0.,
            h=self.h(self.goal),
            parent=None
        )

        heapq.heappush(self.open, (n_.f, n_))

        # for tn, t in self.models.items():
        #     for lidx, l in t.leaves.items():
        #         s_ = State(
        #             tree=tn,
        #             leaf=lidx
        #         )
        #         for var in self.goal.keys:
        #             if var not in l.distributions or var.replace('_out', '_in') not in l.distributions: continue
        #             s_[var] = l.distributions[var.replace('_out', '_in')] + l.distributions[var]
        #
        #         # if s_.posx.p(self.goal.posx) * s_.posy.p(self.goal.posy) >= self.goal_confidence:
        #         if s_.similarity(self.goal) >= self.goal_confidence:
        #             n_ = Node(state=s_, g=0., h=self.h(s_), parent=None)
        #             heapq.heappush(self.open, (n_.f, n_))

    def isgoal(
            self,
            node: Node
    ) -> bool:
        # true, if current belief state is sufficiently similar to init state and ancestor (i.e. first element in parent
        # chain) matches goal specification
        # a = SubAStarBW.get_ancestor(node)
        # return node.state.posx.p(self.initstate.posx.mpe()[1]) * node.state.posy.p(self.initstate.posy.mpe()[1]) >= self.state_similarity and \
        #     a.state.posx.p(self.goal.posx) * a.state.posy.p(self.goal.posy) >= self.goal_confidence
        if not set(self.initstate.keys()).issubset(set([x.replace('_out', '_in') for x in node.state.keys()])): return False
        return reduce(operator.mul, [self.initstate[var].p(node.state[var.replace('_in', '_out')]) for var in self.initstate]) >= self.goal_confidence
            # and a.state.similarity(self.goal) >= self.state_similarity

    @staticmethod
    def get_ancestor(
            node
    ):
        current_node = node
        while current_node.parent is not None:
            current_node = current_node.parent
        return current_node

    def reverse(
            self,
            t: jpt.trees.JPT,
            query: Dict,
            confidence: float = .0
    ) -> Tuple:
        """
        Determines the leaf nodes that match query best and returns them along with their respective confidence.

        :param query: a mapping from featurenames to either numeric value intervals or an iterable of categorical values
        :param confidence:  the confidence level for this MPE inference
        :returns: a tuple of probabilities and jpt.trees.Leaf objects that match requirement (representing path to root)
        """
        # if none of the target variables is present in the query, there is no match possible
        # only check variable names, because multiple trees can have the (semantically) same variable, which differs as
        # python object
        if set([v.name if isinstance(v, Variable) else v for v in query.keys()]).isdisjoint(set(t.varnames)):
            return []

        # Transform into internal values/intervals (symbolic values to their indices)
        query_ = t._preprocess_query(
            query,
            skip_unknown_variables=True
        )

        # update non-query variables to allow all possible values
        for i, var in enumerate(t.variables):
            if var in query_: continue
            if var.numeric:
                query_[var] = R
            else:
                query_[var] = set(var.domain.labels.values())

        # stores the probabilities, that the query variables take on the value(s)/a value in the interval given in
        # the query
        confs = {}

        # find the leaf (or the leaves) that matches the query best
        for k, l in t.leaves.items():
            conf = defaultdict(float)
            for v, dist in l.distributions.items():
                if v.name in query and v.name in l.distributions and v.name.replace('_out', '_in') in l.distributions:
                    if type(l.distributions[v.name]) == Numeric:
                        ndist = Numeric().set(QuantileDistribution.from_cdf(l.distributions[v.name].cdf.xshift(-l.distributions[v.name.replace('_out', '_in')].expectation())))
                    else:
                        ndist = l.distributions[v.name] + l.distributions[v.name.replace('_out', '_in')]
                    newv = ndist.p(query_[v])
                else:
                    newv = dist.p(query_[v])
                conf[v] = newv
            confs[l.idx] = conf

        yield from [(cf, t.leaves[lidx]) for lidx, cf in confs.items() if all(c >= confidence for c in cf.values())]

    def generate_steps(
            self,
            node: Node
    ) -> List[Any]:
        """
        """
        # else:
        #     query = {}
        #     if 'x_in' in node.state.leaf.path:
        #         query['x_in'] = node.state.leaf.distributions['x_in'].mpe()[1]
        #
        #     if 'y_in' in node.state.leaf.path:
        #         query['y_in'] = node.state.leaf.distributions['y_in'].mpe()[1]
        #
        #     if 'xdir_in' in node.state.leaf.path:
        #         query['xdir_in'] = node.state.leaf.distributions['xdir_in'].mpe()[1]
        #
        #     if 'ydir_in' in node.state.leaf.path:
        #         query['ydir_in'] = node.state.leaf.distributions['ydir_in'].mpe()[1]

        query = {
            var: node.state[var] for var in node.state.keys()
        }

        steps = [
            (leaf, treename, tree) for treename, tree in self.models.items() for _, leaf in self.reverse(
                t=tree,
                query=tree.bind(
                    {
                        k: v for k, v in query.items() if k in tree.varnames
                    },
                    allow_singular_values=False
                ),
                confidence=.1  # FIXME: self.goal_confidence?
            )
        ]

        return steps

    def generate_successors(
            self,
            node: Node
    ) -> List[Node]:

        predecessors = []
        for pred, tn, t in self.generate_steps(node):

            # get distributions representing current belief state
            # posx = node.state.posx
            # posy = node.state.posy
            # dirx = node.state.dirx
            # diry = node.state.diry

            # # update new position distributions to reflect the actual (absolute) position after execution
            # if 'x_in' in pred.distributions and 'x_out' in pred.distributions:
            #     posx = Numeric().set(QuantileDistribution.from_cdf(pred.distributions['x_in'].cdf.xshift(-pred.distributions['x_out'].expectation())))
            #
            # if 'y_in' in pred.distributions and 'y_out' in pred.distributions:
            #     posy = Numeric().set(QuantileDistribution.from_cdf(pred.distributions['y_in'].cdf.xshift(-pred.distributions['y_out'].expectation())))
            #
            # # update new orientation distributions to reflect the actual (absolute) direction after execution
            # if 'xdir_in' in pred.distributions and 'xdir_out' in pred.distributions:
            #     dirx = Numeric().set(QuantileDistribution.from_cdf(pred.distributions['xdir_in'].cdf.xshift(-pred.distributions['xdir_out'].expectation())))
            #
            # if 'ydir_in' in pred.distributions and 'ydir_out' in pred.distributions:
            #     diry = Numeric().set(QuantileDistribution.from_cdf(pred.distributions['ydir_in'].cdf.xshift(-pred.distributions['ydir_out'].expectation())))
            #
            # # initialize new belief state for potential predecessor
            # s_ = State(
            #     posx=posx,
            #     posy=posy,
            #     ctree=t,
            #     leaf=pred,
            #     dirx=dirx,
            #     diry=diry,
            #     tn=tn
            # )

            s_ = node.state.__class__()
            s_.update({k: v for k, v in node.state.items()})
            s_['tree'] = tn
            s_['leaf'] = pred.idx

            # initialize new belief state for potential predecessor
            s_.update({var.name.replace('_in', '_out'): pred.distributions[var].value2label(pred.path[var]) for var in pred.path})

            predecessors.append(
                Node(
                    state=s_,
                    g=node.g + self.stepcost(s_),
                    h=self.h(s_),
                    parent=node
                )
            )

        return predecessors
