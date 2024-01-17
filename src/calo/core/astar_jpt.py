import heapq
from collections import defaultdict
from typing import List, Dict, Any, Union

import dnutils
import numpy as np
import pyximport

import jpt
from calo.core.astar import AStar, Node
from calo.utils.constants import calologger
from calo.utils.utils import fmt
from jpt.base.errors import Unsatisfiability
from jpt.distributions import Distribution, Bool, Multinomial, Integer, Numeric
from jpt.variables import Variable
from utils import uniform_numeric

pyximport.install()
from jpt.base.intervals import R, ContinuousSet

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


class Goal(dict):

    def __init__(self, d: dict=None):
        super().__init__()
        self.leaf = None
        self.tree = None
        if d is not None:
            self.update(d)

    def similarity(
            self,
            other: 'Goal'
    ) -> float:
        if not isinstance(other, (Goal, State)): return 0.
        if isinstance(other, Goal):
            # two goals compared with each other must be identical
            if set(self.keys()) != set(other.keys()):
                return 0.
            return np.min(
                [
                    1 if val == other[vn] else 0 for vn, val in self.items()
                ]
            )
        else:
            # a goal and a state are compared in terms of their shared keys by evaluating the goal's values in the
            # distributions of the state
            if set(self.keys()).issubset(set(other.keys())):
                return 0.

            return np.mean(
                [
                    other[vn].pdf(val) for vn, val in self.items()
                ]
            )

    def __str__(self) -> str:
        return f'Goal[{";".join([f"{var}: {fmt(self[var], prec=2)}" for var in self.keys()])}]'

    def __repr__(self) -> str:
        return f'Goal[{", ".join([f"{var}: {fmt(self[var], prec=2)}" for var in self.keys()])}]'

    def __eq__(self, other) -> bool:
        return (self.leaf == other.leaf and
                self.tree == other.tree and
                self.similarity(other) >= 0.8)


class State(dict):

    def __init__(self, d: dict=None):
        super().__init__()
        self.leaf = None
        self.tree = None
        if d is not None:
            self.update(d)

    def similarity(
            self,
            other: 'State'
    ) -> float:
        if not isinstance(other, (Goal, State)): return 0.

        if isinstance(other, Goal):
            # a goal and a state are compared in terms of their shared keys by evaluating the goal's values in the
            # distributions of the state
            if set(other.keys()).issubset(set(self.keys())):
                return 0.

            return np.mean(
                [
                    val.pdf(self[vn]) for vn, val in other.items()
                ]
            )
        else:
            if set(self.keys()) != set(other.keys()):
                return 0.

            # two states are compared by calculating the mean of the single similarities of their shared distributions
            return np.mean(
                [
                    type(val).jaccard_similarity(val, other[vn]) for vn, val in self.items()
                ]
            )

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({", ".join([f"{var}: {fmt(self[var], prec=2)}" for var in self.keys()])})' + \
            f'[{self.tree}({self.leaf})]'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}[{";".join([f"{var}: {fmt(self[var], prec=2)}" for var in self.keys()])}]' + \
            f'[{self.tree}({self.leaf})]'

    def __eq__(self, other) -> bool:
        return (self.leaf == other.leaf and
                self.tree == other.tree and
                self.similarity(other) >= 0.8)


class SubAStar(AStar):

    def __init__(
            self,
            initstate: Any,
            goal: Any,
            models: Dict,
            state_similarity: float = .2,
            goal_confidence: float = .01
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
        logger.debug(f'Init SubAStar...')

        n_ = Node(
            state=self.initstate,
            g=0.,
            h=self.h(self.initstate),
            parent=None
        )

        heapq.heappush(self.open, (n_.f, n_))

    @staticmethod
    def jaccard_similarity(
            d1: ContinuousSet,
            d2: ContinuousSet,
    ) -> float:
        if d1 == d2:
            return 1

        if isinstance(d1, ContinuousSet) and isinstance(d2, ContinuousSet):

            if d1.isinf() or d2.isinf():
                raise ValueError(f"Similarity not defined on infinity intervals. Intervals: d1={d1}, d2={d2} ")

            intersection = d1.intersection(d2).width
            union_ = d1.union(d2)
            if hasattr(union_, "width"):
                union = union_.width
            else:
                union = sum([i.width for i in union_.intervals])
            return intersection / union

        elif isinstance(d1, set) and isinstance(d2, set):
            intersection = d1.intersection(d2)
            union = d1.union(d2)

            # if the union is empty, both sets must be empty, therefore they are identical
            if not union: return 1.
            return len(intersection) / len(union)

        else:
            raise ValueError(f"Both d1 and d2 must be of the same type, either ContinuousSet or set. Got {d1}({type(d1)}) for d1 and {d2}({type(d2)}) for d2.")

    @staticmethod
    def dist(s1, s2):
        # if the states do not share any variables, they are maximally dissimilar
        if not set(s1.keys()).intersection(set(s2.keys())): return np.inf

        # return mean of distances between distributions of common variables in states s1 and s2
        dists = []
        for k in set(s1.keys()).intersection(set(s2.keys())):

            # convert ContinuousSet of variable in state2 into Numeric distribution to allow calculating Wasserstein
            # distance
            if isinstance(s1[k], ContinuousSet):
                v1 = uniform_numeric(s1[k].lower, s1[k].upper)
            else:
                v1 = s1[k]
            if isinstance(s2[k], ContinuousSet):
                v2 = uniform_numeric(s2[k].lower, s2[k].upper)
            else:
                v2 = s2[k]

            # case numeric/integer variables: determine Wasserstein distance
            if isinstance(v1, (Numeric, Integer)):  # this will not work for Integer constraints like {1,2,5,6,7}
                dists.append(Numeric.distance(v1, v2))
            # case multinomial variables: determine 1- probability of matching sets
            elif isinstance(v1, Multinomial):
                # s1 is state
                if isinstance(v2, Multinomial):
                    # s2 is state
                    dists.append(1-v1.pdf(v2.mpe()[0]))
                else:
                    # s2 is goal
                    dists.append(1-v1.pdf(v2))
            elif isinstance(v1, Bool):
                # s1 is state
                if isinstance(v2, Bool):
                    # s2 is state
                    dists.append(1-v1.pdf(v2.mpe()[0]))
                else:
                    # s2 is goal
                    dists.append(1-v1.pdf(v2))
            else:
                # s1 must be Goal (values are sets of multinomial or boolean values)
                if isinstance(v2, (Multinomial, Bool)):
                    # s2 is state
                    dists.append(1-v2.pdf(v1))
                else:
                    # s2 is goal
                    raise ValueError(f"This should never happen. A case where the distance between two Goal objects "
                                     f"has to be calculated, should not occur. Got states {s1} and {s2}")
                    # dists.append(0 if v1 == v2 else 1)

        return np.mean(dists)

    def isgoal(
            self,
            node: Node
    ) -> bool:
        # if not all required fields of the goal are contained in the state of the current node, it cannot match
        # the goal specification
        if not set(self.goal.keys()).issubset(set(node.state.keys())): return False

        for var in self.goal:
            if isinstance(node.state[var], Distribution):
                # case node state is intermediate state -> values are distributions
                # if isinstance(self.goal[var], ContinuousSet):
                intersection = node.state[var].mpe()[0].intersection(self.goal[var])
                if self.jaccard_similarity(intersection, node.state[var].mpe()[0]) < .7:
                    return False
            else:
                # case node state is Goal -> values are (continuous) sets
                if self.jaccard_similarity(node.state[var], self.goal[var]) < .7:
                    return False

        return True

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
            var: ContinuousSet(node.state[var].ppf(.05), node.state[var].ppf(.95)) if hasattr(node.state[var], 'ppf') else node.state[var].mpe()[0] for var in node.state.keys()
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

        return condtrees

        # posteriors = [
        #     [
        #         tn,
        #         tree.posterior(
        #             variables=tree.targets,
        #             evidence=tree.bind(
        #                 {k: v for k, v in evidence.items() if k in tree.varnames},
        #                 allow_singular_values=False
        #             ),
        #             fail_on_unsatisfiability=False
        #         )
        #     ] for tn, tree in self.models.items()
        # ]

        # return posteriors

    def generate_successors(
            self,
            node
    ) -> List[Node]:

        successors = []
        # for best, tn, t in self.generate_steps(node):
        for tn, condtree in self.generate_steps(node):
            if condtree is None: continue

            # each leaf of the conditional tree poses a potential candidate for a successor state
            for idx, leaf in condtree.leaves.items():

                # copy previous state
                s_ = self.state_t()
                s_.update({k: v for k, v in node.state.items()})

                # update tree and leaf that represent this step
                s_.tree = tn
                s_.leaf = idx

                # update belief state of potential predecessor
                for vn, d in leaf.distributions.items():
                    vname = vn.name
                    outvar = vn.name.replace('_in', '_out')
                    invar = vn.name.replace('_out', '_in')

                    # update belief state of potential predecessor for vn, d in best.distributions.items():
                    if vname.endswith('_out') and vname.replace('_out', '_in') in s_:
                        # if the _in variable is already contained in the state, update it by adding the delta
                        # from the leaf distribution
                        indist = s_[invar]
                        outdist = leaf.distributions[outvar]
                        if len(indist.cdf.functions) > 20:
                            # print(f"A Approximating {invar} distribution of s_ with {len(indist.cdf.functions)} functions")
                            indist = indist.approximate(n_segments=20)
                        if len(outdist.cdf.functions) > 20:
                            # print(f"B Approximating {outvar} distribution of best with {len(outdist.cdf.functions)} functions")
                            outdist = outdist.approximate(n_segments=20)
                        vname = invar
                        s_[vname] = indist + outdist
                    elif vname.endswith('_in') and vname in s_:
                        # do not overwrite '_in' distributions
                        continue
                    else:
                        s_[vname] = d

                    if hasattr(s_[vname], 'approximate'):
                        # print(f"C Approximating {vname} distribution of s_ (result) with {len(s_[vname].cdf.functions)} functions")
                        s_[vname] = s_[vname].approximate(n_segments=20)

                successors.append(
                    Node(
                        state=s_,
                        g=node.g + self.stepcost(s_, node.state),
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
            state_similarity: float = .2,
            goal_confidence: float = .2
    ):
        super().__init__(
            initstate,
            goal,
            models=models,
            state_similarity=state_similarity,
            goal_confidence=goal_confidence
        )

    def init(self):
        logger.debug(f'Init SubAStarBW...')
        # node from goalstate
        n_ = Node(
            state=self.goal,
            g=0.,
            h=self.h(self.goal),
            parent=None
        )

        heapq.heappush(self.open, (n_.f, n_))


    @staticmethod
    def jaccard_similarity(
            d1: Union[ContinuousSet, set],
            d2: Union[ContinuousSet, set],
    ) -> float:
        if d1 == d2:
            return 1

        if isinstance(d1, ContinuousSet) and isinstance(d2, ContinuousSet):

            if d1.isinf() or d2.isinf():
                raise ValueError(f"Similarity not defined on infinity intervals. Intervals: d1={d1}, d2={d2} ")

            intersection = d1.intersection(d2).width
            union_ = d1.union(d2)
            if hasattr(union_, "width"):
                union = union_.width
            else:
                union = sum([i.width for i in union_.intervals])

            # if the union is empty, both sets must be empty, therefore they are identical
            if not union: return 1.
            return intersection / union

        elif isinstance(d1, set) and isinstance(d2, set):
            intersection = d1.intersection(d2)
            union = d1.union(d2)

            # if the union is empty, both sets must be empty, therefore they are identical
            if not union: return 1.
            return len(intersection) / len(union)

        else:
            raise ValueError(f"Both d1 and d2 must be of the same type, either ContinuousSet or set. Got {d1}({type(d1)}) for d1 and {d2}({type(d2)}) for d2.")

    @staticmethod
    def dist(s1, s2):
        # if the states do not share any variables, they are maximally dissimilar
        if not set(s1.keys()).intersection(set(s2.keys())): return np.inf

        # return mean of distances between distributions of common variables in states s1 and s2
        dists = []
        for k in set(s1.keys()).intersection(set(s2.keys())):

            # convert ContinuousSet of variable in state2 into Numeric distribution to allow calculating Wasserstein
            # distance
            if isinstance(s1[k], ContinuousSet):
                v1 = uniform_numeric(s1[k].lower, s1[k].upper)
            else:
                v1 = s1[k]
            if isinstance(s2[k], ContinuousSet):
                v2 = uniform_numeric(s2[k].lower, s2[k].upper)
            else:
                v2 = s2[k]

            # case numeric/integer variables: determine Wasserstein distance
            if isinstance(v1, (Numeric, Integer)):  # this will not work for Integer constraints like {1,2,5,6,7}
                dists.append(Numeric.distance(v1, v2))
            # case multinomial variables: determine 1- probability of matching sets
            elif isinstance(v1, Multinomial):
                # s1 is state
                if isinstance(v2, Multinomial):
                    # s2 is state
                    dists.append(1-v1.pdf(v2.mpe()[0]))
                else:
                    # s2 is goal
                    dists.append(1-v1.pdf(v2))
            elif isinstance(v1, Bool):
                # s1 is state
                if isinstance(v2, Bool):
                    # s2 is state
                    dists.append(1-v1.pdf(v2.mpe()[0]))
                else:
                    # s2 is goal
                    dists.append(1-v1.pdf(v2))
            else:
                # s1 must be Goal (values are sets of multinomial or boolean values)
                if isinstance(v2, (Multinomial, Bool)):
                    # s2 is state
                    dists.append(1-v2.pdf(v1))
                else:
                    # s2 is goal
                    raise ValueError(f"This should never happen. A case where the distance between two Goal objects "
                                     f"has to be calculated, should not occur. Got states {s1} and {s2}")
                    # dists.append(0 if v1 == v2 else 1)

        return np.mean(dists)

    def isgoal(
            self,
            node: Node
    ) -> bool:
        # # if not all required fields of the initstate are contained in the state of the current node, it cannot match
        # # this will already fail for the first step in most cases, in which `node` is the goal state
        # REMOVED BECAUSE: initstate does not necessarily have to be a complete assignment of all possible variables,
        # as these may not be known beforehand. It is much more important, that the variables that ARE present in
        # the initstate do not violate the requirements (preconditions) of the current state
        # if not set(node.state.keys()).issubset(set(self.initstate.keys())): return False

        if self.dist(node.state, self.initstate) <= 0.11: return True

        for var, val in node.state.items():
            if var not in self.initstate.keys(): continue  # skip check for variables not present in the initstate
            if isinstance(val, Distribution):
                # default: node.state is belief state (values of both node.state and self.initstate are distributions)
                # if node.state[var].p(self.initstate[var].mpe()[0]) < .7 and type(node.state[var]).jaccard_similarity(node.state[var], self.initstate[var]) < .8:
                if type(val).jaccard_similarity(self.initstate[var], val) < .7:
                    return False
            else:
                # first step: node.state is Goal object (values are sets or ContinuousSets)
                # if less than 70% of the node.state match the goal spec, return false
                intersection = val.intersection(self.initstate[var].mpe()[0])
                if self.jaccard_similarity(intersection, val) < .7:
                    return False
        return True

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
            treename: str = None,
    ) -> List:
        """
        Determines the leaf nodes that match query best and returns them along with their respective confidence.

        :param query: a mapping from featurenames to either numeric value intervals or an iterable of categorical values
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
            s_ = self.state_t()
            s_.tree = treename
            s_.leaf = l.idx

            for v, _ in l.distributions.items():
                vname = v.name
                invar = vname.replace('_out', '_in')
                outvar = vname.replace('_in', '_out')

                if vname.endswith('_in') and vname.replace('_in', '_out') in l.distributions:
                    # if the current variable is an _in variable, and contains the respective _out variable
                    # distribution, add the two distributions and calculate probability on resulting
                    # distribution
                    outdist = l.distributions[outvar]
                    indist = l.distributions[invar]

                    # determine probability that this action (leaf) produces desired output for this variable
                    tmp_dist = indist + outdist
                    c_ = tmp_dist.p(query_[vname])

                    # check here for stopping criterion to save expensive calculations below
                    if not c_ > 0:
                        return

                    # determine distribution from which the execution of this action (leaf) produces desired output
                    try:
                        cond = tmp_dist.crop(query_[vname])
                        tmp_diff = cond - outdist
                        d_ = tmp_diff.approximate(n_segments=20)
                    except Unsatisfiability:
                        return
                elif vname.endswith('_out'):
                    # do not write out variables into belief state
                    continue
                else:
                    # default case
                    c_ = l.distributions[vname].p(query_[vname])
                    d_ = l.distributions[vname]

                # drop entire leaf as soon as one variable has probability 0
                if not c_ > 0:
                    return
                conf[vname] = c_
                s_[vname] = d_
            return s_, conf

        # find the leaf (or the leaves) that matches the query best
        steps = []
        for i, (k, l) in enumerate(t.leaves.items()):
            res = determine_leaf_confs(l)
            if res is not None:
                steps.append(res)

        return steps

    def generate_steps(
            self,
            node: Node
    ) -> List[Any]:
        """
        """
        # ascertain in generate_successors, that node.state only contains _out variables
        query = {
            var:
                node.state[var] if isinstance(node.state[var], (set, ContinuousSet)) else
                node.state[var].mpe()[0] for var in node.state.keys()
        }

        # steps contains triples of (state, confidence, leaf prior) generated by self.reverse
        steps = []
        for treename, tree in self.models.items():
            steps.extend(
                self.reverse(
                    t=tree,
                    treename=treename,
                    query=query
                )
            )

        # sort candidates according to overall confidence (=probability to reach) and select n best ones
        n = 50
        selected_steps = sorted(steps, reverse=True, key=lambda x: np.product([v for _, v in x[1].items()]))[:n]

        # add info so selected_steps contains tuples (step, confidence, distance to init state, leaf prior)
        selected_steps = [(s, c, self.h(s)) for s, c in selected_steps]

        # sort remaining candidates according to distance to init state (=goal in reverse)
        # selected_steps = sorted(selected_steps, key=lambda x: x[2])

        return selected_steps

    def generate_successors(
            self,
            node: Node
    ) -> List[Node]:

        predecessors = []
        for s_, conf, dist in self.generate_steps(node):

            predecessors.append(
                Node(
                    state=s_,
                    g=node.g + self.stepcost(node.state, s_),
                    h=dist,  # self.h(s_),  # do not calculate distance twice
                    parent=node
                )
            )

        return predecessors
