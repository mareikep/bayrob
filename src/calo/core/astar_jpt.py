import heapq
import heapq
import operator
from collections import defaultdict
from functools import reduce
from typing import List, Dict, Any, Tuple

import dnutils
import pyximport
from jpt.distributions.quantile.quantiles import QuantileDistribution

import jpt
from calo.core.astar import AStar, Node
from calo.utils.constants import calologger, nl, cst
from calo.utils.utils import fmt
from jpt.distributions import Numeric, Distribution, Integer
from jpt.variables import Variable

pyximport.install()
from jpt.base.intervals import R, ContinuousSet

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


class Goal(dict):

    def __init__(self):
        self.leaf = None
        self.tree = None

    def similarity(
            self,
            other: 'Goal'
    ) -> float:
        if not isinstance(other, Goal): return False

        if set(self.keys()) != set(other.keys()):
            raise ValueError('Variable sets do not match.')

        return min(
            [
                1 if self[vn] == other[vn] else 0 for vn, val in self.items()
            ]
        )

    def __str__(self) -> str:
        return f'GOAL[{";".join([f"{var}: {fmt(self[var], prec=2)}" for var in self.keys()])}]'

    def __repr__(self) -> str:
        return f'Goal[{", ".join([f"{var}: {fmt(self[var], prec=2)}" for var in self.keys()])}]'


class State(dict):

    def __init__(self):
        self.leaf = None
        self.tree = None
        super().__init__()

    def similarity(
            self,
            other: 'State'
    ) -> float:
        if not isinstance(other, State): return False

        if set(self.keys()) != set(other.keys()):
            return 0#raise ValueError('Variable sets do not match.')

        return min(
            [
                type(self[vn]).jaccard_similarity(val, other[vn]) for vn, val in self.items()
            ]
        )

    def __str__(self) -> str:
        return f'{self.__class__.__name__.upper()}({";".join([f"{var}: {fmt(self[var], prec=2)}" for var in self.keys()])})' + \
            f'[{self.tree}({self.leaf})]'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}[{", ".join([f"{var}: {fmt(self[var], prec=2)}" for var in self.keys()])}]' + \
            f'[{self.tree}({self.leaf})]'


class SubAStar(AStar):

    def __init__(
            self,
            initstate: State,
            goal: Goal,
            models: Dict,
            state_similarity: float = .9,
            goal_confidence: float = 1
    ):
        self.models = models
        self.state_t = type(initstate)
        self.goal_t = type(goal)
        super().__init__(
            initstate,
            goal,
            state_similarity=state_similarity,
            goal_confidence=goal_confidence
        )

    def init(self):
        n_ = Node(
            state=self.initstate,
            g=0.,
            h=self.h(self.initstate),
            parent=None
        )

        heapq.heappush(self.open, (n_.f, n_))

    def isgoal(
            self,
            node: Node
    ) -> bool:
        # if not all required fields of the goal are contained in the state of the current node, it cannot match
        # the goal specification
        if not set([k.replace('_out', '_in') for k in self.goal.keys()]).issubset(set(node.state.keys())): return False

        # otherwise, return whether probability that the current state matches the goal specification is greater or
        # equal to the user-defined goal-confidence, i.e. return true, if current belief state is sufficiently
        # similar to goal
        return reduce(operator.mul, [node.state[var.replace('_out', '_in')].p(self.goal[var]) if isinstance(node.state, Distribution) else 1 if node.state[var.replace('_out', '_in')].mpe()[1] == self.goal[var] else 0 for var in self.goal]) >= self.goal_confidence

    def generate_steps(
            self,
            node
    ) -> List[Any]:
        """Generates potential next steps by restricting the trees to only contain leaves that are reachable from the
        current position.

        :param node: the current node
        :type node: SubNode
        """
        # generate evidence by using intervals from the 5th percentile to the 95th percentile for each distribution
        # TODO: remove else case once ppf exists for Integer
        evidence = {
            var: ContinuousSet(node.state[var].ppf(.05), node.state[var].ppf(.95)) if hasattr(node.state[var], 'ppf') else node.state[var].mpe()[1] for var in node.state.keys()
            # var: node.state[var].mpe()[1]  # for Integer variables
        }

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

            # copy previous state
            s_ = self.state_t()
            s_.update({k: v for k, v in node.state.items()})

            # update tree and leaf that represent this step
            s_.tree = tn
            s_.leaf = succ.idx

            # update belief state of potential predecessor
            for vn, d in succ.distributions.items():
                # generate new distribution adding position delta distribution to original dist
                # belief state
                if vn.name != vn.name.replace('_in', '_out') and vn.name.replace('_in', '_out') in succ.distributions:

                    if vn.name in s_:
                        # if the _in variable is already contained in the state, update it by adding the delta
                        # from the leaf distribution
                        nsegments = len(s_[vn.name].cdf.functions)
                        s_[vn.name] = s_[vn.name] + succ.distributions[vn.name.replace('_in', '_out')]
                    else:
                        # else save the result of the _in from the leaf distribution shifted by its delta (_out)
                        nsegments = len(d.pdf.functions)
                        s_[vn.name] = d + succ.distributions[vn.name.replace('_in', '_out')]

                    # reduce complexity from adding two distributions to complexity of previous, unaltered distribution
                    # TODO: remove condition once ppf exists for Integer
                    if hasattr(s_[vn.name], 'approximate'):
                        nsegments = min(10, nsegments)
                        s_[vn.name] = s_[vn.name].approximate(
                            n_segments=nsegments,
                            # error_max=.1
                        )

            successors.append(
                Node(
                    state=s_,
                    g=node.g + self.stepcost(s_),
                    h=self.h(s_),
                    parent=node
                )
            )
        return successors


class SubAStarBW(SubAStar):

    def __init__(
            self,
            initstate: State,  # would be the goal state of forward-search
            goal: Goal,  # init state in forward-search
            models: Dict,
            state_similarity: float = .9,
            goal_confidence: float = 1
    ):
        super().__init__(
            initstate,
            goal,
            models=models,
            state_similarity=state_similarity,
            goal_confidence=goal_confidence
        )

    def init(self):
        # node from goalstate
        n_ = Node(
            state=self.goal,
            g=0.,
            h=self.h(self.goal),
            parent=None
        )

        heapq.heappush(self.open, (n_.f, n_))

    def isgoal(
            self,
            node: Node
    ) -> bool:
        # true, if current belief state is sufficiently similar to init state and ancestor (i.e. first element in parent
        # chain) matches goal specification
        # a = SubAStarBW.get_ancestor(node)  # TODO: this should be

        # if not all required fields of the initstate are contained in the state of the current node, it does not match
        if not set(self.initstate.keys()).issubset(set(node.state.keys())): return False
        # if not set(self.goal.keys()).issubset(set(a.state.keys())): return False

        # otherwise, return the probability that the current state matches the initial state TODO: greater equal state similarity??
        return reduce(operator.mul, [self.initstate[var].p(node.state[var].mpe()[1] if isinstance(node.state[var], Distribution) else node.state[var]) for var in self.initstate]) >= self.state_similarity #\
            # and reduce(operator.mul, [a.state[var].p(self.goal[var]) for var in self.goal]) >= self.goal_confidence


    @staticmethod
    def get_ancestor(
            node
    ):
        current_node = node
        while current_node.parent is not None and not isinstance(current_node.parent.state, Goal):
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

        q_ = t.bind(
            {
                k: v for k, v in query.items() if k in t.varnames
            },
            allow_singular_values=False
        )

        # Transform into internal values/intervals (symbolic values to their indices)
        query_ = t._preprocess_query(
            q_,
            skip_unknown_variables=True
        )

        # update non-query variables to allow all possible values
        for i, var in enumerate(t.variables):
            if var in query_: continue
            if var.numeric:
                query_[var] = R
            else:
                query_[var] = set(var.domain.labels.values())

        def determine_leaf_confs(l):
            #  assuming, that the _out variables are deltas but querying for _out semantically means querying for the
            #  result of adding the delta to the _in variable (i.e. the actual outcome of performing the action
            #  represented by the leaf)
            conf = defaultdict(float)
            for v, dist in l.distributions.items():
                vname_in = v.name.replace('_out', '_in')
                vname_out = v.name.replace('_in', '_out')

                # if the current variable is an _out variable and its corresponding _in variable exists in the
                # distribution, add the two distributions and calculate probability on resulting distribution
                if v.name != vname_in and v.name in query and vname_in in l.distributions:
                    ndist = dist + l.distributions[vname_in]
                    c_ = ndist.p(query_[v])
                else:
                    c_ = dist.p(query_[v])
                if c_ < confidence: return
                conf[v.name] = c_

                # if the current variable is an _in variable, and it has no corresponding _out variable but is part of
                # the original query, it is considered to be left unchanged by the "execution" of the action
                # represented by the leaf.
                # Therefore, the distribution of the input variable is taken as basis for calculating the
                # probability
                # Note: check if vname_out in query but take value from query_, as this is preprocessed and MIGHT differ
                if v.name != vname_out and vname_out in query and not vname_out in l.distributions:
                    c_ = dist.p(query[vname_out])
                    if c_ < confidence: return
                    conf[vname_out] = c_
            return conf

        # find the leaf (or the leaves) that matches the query best
        for k, l in t.leaves.items():
            conf = determine_leaf_confs(l)
            if conf is None: continue
            yield conf, l

    def generate_steps(
            self,
            node: Node
    ) -> List[Any]:
        """
        """
        # ascertain in generate_successors, that node.state only contains _out variables
        query = {
            var.replace('_in', '_out'):
                node.state[var] if isinstance(node.state[var], (set, ContinuousSet)) else
                node.state[var].mpe()[1] for var in node.state.keys()
        }

        steps = [
            (leaf, treename, tree) for treename, tree in self.models.items() for _, leaf in self.reverse(
                t=tree,
                query=query,
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

            # copy previous state
            s_ = self.state_t()
            s_.update({k: v for k, v in node.state.items()})

            # update tree and leaf that represent this step
            s_.tree = tn
            s_.leaf = pred.idx

            # update belief state of potential predecessor
            for v, d in pred.distributions.items():
                if v not in t.features: continue  # TODO: take all FEATURE variables, not only _in ones
                s_[v.name] = d

            predecessors.append(
                Node(
                    state=s_,
                    g=node.g + self.stepcost(s_),
                    h=self.h(s_),
                    parent=node
                )
            )

        return predecessors
