import heapq
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import dnutils
import numpy as np
import pyximport

import jpt
from calo.core.astar import AStar, Node
from calo.utils.constants import calologger
from calo.utils.utils import fmt
from jpt.base.errors import Unsatisfiability
from jpt.distributions import Distribution, Bool, Multinomial
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
    def dist(s1, s2):
        # return mean of distances between respective distributions in states s1 and s2
        dists = []
        for k, v in s2.items():
            if k in s1:
                from jpt.distributions import Numeric
                dists.append(Numeric.distance(s1[k], v))
        return np.mean(dists)

    def isgoal(
            self,
            node: Node
    ) -> bool:
        # if not all required fields of the goal are contained in the state of the current node, it cannot match
        # the goal specification
        if not set(self.goal.keys()).issubset(set(node.state.keys())): return False

        vars_valid = []
        for var in self.goal:
            if isinstance(node.state[var], Distribution):
                if isinstance(self.goal[var], (set, ContinuousSet)):
                    # case var has numeric distribution
                    vars_valid.append(node.state[var].mpe()[0] in self.goal[var])
                else:
                    # case var has multinomial distribution
                    vars_valid.append(node.state[var].mpe()[0] == self.goal[var])
            else:
                # case var is anything else
                vars_valid.append(node.state[var] == self.goal[var])

        return all(vars_valid)

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
                tree,
                tree.conditional_jpt(
                    evidence=tree.bind(
                        {k: v for k, v in evidence.items() if k in tree.varnames},
                        allow_singular_values=False
                    ),
                    fail_on_unsatisfiability=False)
            ] for tn, tree in self.models.items()
        ]
        posteriors = [
            [
                tn,
                c.posterior(
                    variables=tree.targets
                )
            ] for tn, tree, c in condtrees
        ]

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

        return posteriors

    def generate_successors(
            self,
            node
    ) -> List[Node]:

        successors = []
        # for best, tn, t in self.generate_steps(node):
        for tn, best in self.generate_steps(node):

            # copy previous state
            s_ = self.state_t()
            s_.update({k: v for k, v in node.state.items()})

            # update tree and leaf that represent this step
            s_.tree = tn
            s_.leaf = None  # best.idx

            # update belief state of potential predecessor
            for vn, d in best.items():
                vname = vn.name
                outvar = vn.name.replace('_in', '_out')
                invar = vn.name.replace('_out', '_in')

                # update belief state of potential predecessor for vn, d in best.distributions.items():
                if vname.endswith('_out') and vname.replace('_out', '_in') in s_:
                    # if the _in variable is already contained in the state, update it by adding the delta
                    # from the leaf distribution
                    indist = s_[invar]
                    outdist = best[outvar]
                    if len(indist.cdf.functions) > 20:
                        print(f"A Approximating {invar} distribution of s_ with {len(indist.cdf.functions)} functions")
                        indist = indist.approximate(n_segments=20)
                    if len(outdist.cdf.functions) > 20:
                        print(
                            f"B Approximating {outvar} distribution of best with {len(outdist.cdf.functions)} functions")
                        outdist = outdist.approximate(n_segments=20)
                    vname = invar
                    s_[vname] = indist + outdist
                elif vname.endswith('_in') and vname in s_:
                    # do not overwrite '_in' distributions
                    continue
                else:
                    s_[vname] = d

                if hasattr(s_[vname], 'approximate'):
                    print(f"C Approximating {vname} distribution of s_ (result) with {len(s_[vname].cdf.functions)} functions")
                    s_[vname] = s_[vname].approximate(n_segments=20)

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

    def isgoal(
            self,
            node: Node
    ) -> bool:

        # if not all required fields of the initstate are contained in the state of the current node, it does not match
        if not set(self.initstate.keys()).issubset(set(node.state.keys())): return False

        sims = []
        for var in self.initstate:
            if isinstance(node.state[var], Distribution):
                # any intermediate state should be Distribution
                sims.append(self.initstate[var].similarity(node.state[var]))
            else:
                # otherwise it must be the defined Goal in Step 0
                sims.append(1 if self.initstate[var].mpe()[0] in node.state[var] else 0)

        return np.mean(sims) >= self.state_similarity

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

        logger.warning(f"Tree {treename} QUERY: \n{query}\n{query_}")

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

                    # determine distribution from which the execution of this action (leaf) produces desired output
                    try:
                        cond = tmp_dist.crop(query_[vname])
                        tmp_diff = cond - outdist
                        tmp_diff = tmp_dist.approximate(n_segments=20)
                        d_ = tmp_diff
                    except Unsatisfiability:
                        c_ = 0
                else:
                    # default case
                    c_ = l.distributions[vname].p(query_[vname])
                    d_ = l.distributions[vname]

                # drop entire leaf if only one variable has probability 0
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

        # sort candidates according to overall confidence (=probability to reach) and select 10 best ones
        selected_steps = sorted(steps, reverse=True, key=lambda x: np.product([v for _, v in x[1].items()]))[:10]

        # add info so selected_steps contains tuples (step, confidence, distance to init state, leaf prior)
        selected_steps_ = [(s, c, self.h(s)) for s, c in selected_steps]

        # sort remaining candidates according to distance to init state (=goal in reverse)
        selected_steps_wasserstein = sorted(selected_steps_, key=lambda x: x[2])

        return selected_steps_wasserstein

    def generate_successors(
            self,
            node: Node
    ) -> List[Node]:

        predecessors = []
        for s_, conf, dist in self.generate_steps(node):

            predecessors.append(
                Node(
                    state=s_,
                    g=node.g + self.stepcost(s_),
                    h=dist,  # self.h(s_),  # do not calculate distance twice
                    parent=node
                )
            )

        return predecessors
