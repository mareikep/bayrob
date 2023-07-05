import datetime
import heapq
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple

import jpt
import numpy as np
from jpt.distributions.quantile.quantiles import QuantileDistribution

import dnutils
import pyximport
from calo.core.astar import AStar, Node, BiDirAStar
from calo.logs.logs import init_loggers
from calo.models.action import Move
from calo.utils import locs
from calo.utils.constants import calologger, plotcolormap, FILESTRFMT_SEC
from calo.utils.utils import pnt2line, recent_example, angledeg
from dnutils import ifnone, first
from jpt.distributions import Numeric, Gaussian
from jpt.trees import JPT
from jpt.variables import Variable
from matplotlib import pyplot as plt

pyximport.install()
from jpt.base.intervals import ContinuousSet, R

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


class Goal:
    def __init__(
            self,
            posx: Union[float, ContinuousSet],
            posy: Union[float, ContinuousSet]
    ):
        self.posx = posx
        self.posy = posy

    def __str__(self) -> str:
        return f'<State pos: ({str(self.posx)}/{str(self.posy)})>'

    def __repr__(self) -> str:
        return str(self)


class State:
    def __init__(
            self,
            posx: Numeric,
            posy: Numeric,
            dirx: Numeric = None,
            diry: Numeric = None,
            ctree: Any = None,
            leaf: Any = None,
            tn: str = None
    ):
        self.posx = posx
        self.posy = posy
        self.dirx = dirx
        self.diry = diry
        self.ctree = ctree
        self.leaf = leaf
        self.tn = tn

    def __eq__(
            self,
            other
    ):
        return self.posx == other.posx and self.posy == other.posy

    def __str__(self):
        posx = self.posx.expectation()
        posy = self.posy.expectation()
        if self.dirx is not None and self.diry is not None:
            dirx = self.dirx.expectation()
            diry = self.diry.expectation()
            dirxy = f"; dir: ({dirx:.2f}/{diry:.2f})"
        else:
            dirxy = ""

        return f'<State pos: ({posx:.2f}/{posy:.2f}){dirxy}>'

    def __repr__(self):
        if self.tn is None or self.leaf is None:
            return "Init"
        return f"{self.tn}({ifnone(self.leaf, '', lambda _: self.leaf.idx)})"

    def similarity(
            self,
            other: 'State'
    ) -> float:
        return min(
            [
                Numeric.jaccard_similarity(self.posx, other.posx),
                Numeric.jaccard_similarity(self.posy, other.posy),
                Numeric.jaccard_similarity(self.dirx, other.dirx),
                Numeric.jaccard_similarity(self.diry, other.diry),
            ]
        )

    def plot(
            self,
            title: str = None,
            conf: float = None,
            limx: Tuple = None,
            limy: Tuple = None,
            limz: Tuple = None,
            save: str = None,
            show: bool = True
    ) -> None:
        """Plots a heatmap representing the belief state for the agents' position, i.e. the joint
        probability of the x and y variables: P(x, y).

        :param title: The plot title
        :param conf:  A confidence value. Values below this threshold are set to 0. (= equal color for lowest value in plot)
        :param limx: The limits for the x-variable; determined from boundaries if not given
        :param limy: The limits for the y-variable; determined from boundaries if not given
        :param limz: The limits for the z-variable; determined from data if not given
        :param save: The location where the plot is saved (if given)
        :param show: Whether the plot is shown
        :return: None
        """
        # generate datapoints
        x = self.posx.pdf.boundaries()
        y = self.posy.pdf.boundaries()

        # determine limits
        xmin = ifnone(limx, min(x) - 15, lambda l: l[0])
        xmax = ifnone(limx, max(x) + 15, lambda l: l[1])
        ymin = ifnone(limy, min(y) - 15, lambda l: l[0])
        ymax = ifnone(limy, max(y) + 15, lambda l: l[1])

        X, Y = np.meshgrid(x, y)
        Z = np.array(
            [
                self.posx.pdf(x) * self.posy.pdf(y)
                for x, y, in zip(X.ravel(), Y.ravel())
            ]).reshape(X.shape)

        # show only values above a certain threshold, consider lower values as high-uncertainty areas
        if conf is not None:
            Z[Z < conf] = 0.

        # remove or replace by eliminating values > median
        Z[Z > np.median(Z)] = np.median(Z)

        zmin = ifnone(limz, Z.min(), lambda l: l[0])
        zmax = ifnone(limz, Z.max(), lambda l: l[1])

        # init plot
        fig, ax = plt.subplots(num=1, clear=True)
        fig.patch.set_facecolor('#D6E7F8')  # set bg color around the plot area (royal purple)
        ax.set_facecolor('white')  # set bg color of plot area (dark purple)
        cmap = 'BuPu'  # viridis, Blues, PuBu, 0rRd, BuPu

        # generate heatmap
        c = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=zmin, vmax=zmax)
        ax.set_title(f'P(x,y)')

        # setting the limits of the plot to the limits of the data
        ax.axis([xmin, xmax, ymin, ymax])
        # ax.axis([-100, 100, -100, 100])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        fig.colorbar(c, ax=ax)
        fig.suptitle(title)
        fig.canvas.manager.set_window_title(f'Belief State: P(x/y)')

        if save:
            plt.savefig(save)

        if show:
            plt.show()


class SubAStar(AStar):

    def __init__(
            self,
            initstate: State,
            goalstate: Goal,  # might be belief state later
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

    def stepcost(
            self,
            state
    ) -> float:
        # distance (Euclidean) travelled so far (from init_pos to current position)
        cost = 0.
        dx = self.initstate.posx.expectation() - state.posx.expectation()
        dy = self.initstate.posy.expectation() - state.posy.expectation()
        cost += math.sqrt(dx ** 2 + dy ** 2)

        # difference in orientation (from init dir to current dir)
        # cost += angledeg([state.dirx, state.diry], [self.initstate.dirx, self.initstate.diry])
        if 'angle' in state.leaf.value:
            cost += abs(state.leaf.value['angle'].expectation())

        return cost

    def h(
            self,
            state: State
    ) -> float:
        p = 1
        if not any([x is None for x in [state.posx, state.posy]]):
            p *= state.posx.p(self.goalstate.posx) * state.posy.p(self.goalstate.posy)

        # if not any([x is None for x in [state.dirx, state.diry]]):
        #     # determine direction vector to nearest point of goal area
        #     d, pt = pnt2line([state.posx.expectation(), state.posy.expectation()], [self.goalstate.posx.lower, self.goalstate.posy.lower], [self.goalstate.posx.upper, self.goalstate.posy.upper])
        #     dx_ = pt[0] - state.posx.expectation()
        #     dy_ = pt[1] - state.posy.expectation()
        #
        #     # normalize direction vector
        #     l = math.sqrt(dx_**2 + dy_**2)
        #     dx_ /= l
        #     dy_ /= l
        #
        #     # add noise to prevent 0-probabilities
        #     tolerance = .2
        #     dx = ContinuousSet(dx_ - abs(tolerance * dx_), dx_ + abs(tolerance * dx_))
        #     dy = ContinuousSet(dy_ - abs(tolerance * dy_), dy_ + abs(tolerance * dy_))
        #     p *= state.dirx.p(dx) * state.diry.p(dy)

        return 1 - p

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
            'x_in': node.state.posx.mpe()[1],
            'y_in': node.state.posy.mpe()[1],
            'xdir_in': node.state.dirx.mpe()[1],
            'ydir_in': node.state.diry.mpe()[1]
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

        # for debugging, TODO: remove!
        # for i, jpt_ in condtrees:
        #     if jpt_ is None: continue
            # jpt_.plot(
            #     title=i,
            #     plotvars=jpt_.variables,
            #     leaffill='#CCDAFF',
            #     nodefill='#768ABE',
            #     alphabet=True
            # )
            # if i == 'MOVEFORWARD.tree' and jpt_:
            #     Move.plot(
            #         jpt_=jpt_,
            #         qvarx=jpt_.varnames['x_out'],
            #         qvary=jpt_.varnames['y_out'],
            #         evidence={jpt_.varnames[k]: v for k, v in evidence.items()},
            #         title=f'{i} (conditional)',
            #         # conf=.0003,
            #         limx=(-150, 150),
            #         limy=(-150, 150),
            #         # limz=(0, 0.001),
            #         save=os.path.join(locs.logs, f'{i}_cond-{datetime.datetime.now().strftime(FILESTRFMT_SEC)}.png'),
            #         show=False
            #     )

        return [(leaf, treename, tree) for treename, tree in condtrees if tree is not None for _, leaf in tree.leaves.items()]

    def generate_successors(
            self,
            node
    ) -> List[Node]:
        successors = []
        for succ, tn, t in self.generate_steps(node):

            # get distributions representing current belief state
            posx = node.state.posx
            posy = node.state.posy
            dirx = node.state.dirx
            diry = node.state.diry

            # generate new position distribution by shifting position delta distributions by expectation of position
            # belief state
            if 'x_out' in succ.distributions:
                posx = Numeric().set(QuantileDistribution.from_cdf(succ.value['x_out'].cdf.xshift(-posx.expectation())))

            if 'y_out' in succ.distributions:
                posy = Numeric().set(QuantileDistribution.from_cdf(succ.value['y_out'].cdf.xshift(-posy.expectation())))

            # generate new orientation distribution by shifting orientation delta distributions by expectation of
            # orientation belief state
            if 'xdir_out' in succ.distributions:
                dirx = Numeric().set(QuantileDistribution.from_cdf(succ.value['xdir_out'].cdf.xshift(-dirx.expectation())))

            if 'ydir_out' in succ.distributions:
                diry = Numeric().set(QuantileDistribution.from_cdf(succ.value['ydir_out'].cdf.xshift(-diry.expectation())))

            # initialize new belief state for potential successor
            state = State(
                posx=posx,
                posy=posy,
                ctree=t,
                leaf=succ,
                dirx=dirx,
                diry=diry,
                tn=tn
            )

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
        return node.state.posx.p(self.goalstate.posx) * node.state.posy.p(self.goalstate.posy) >= self.goal_confidence

    def plot(
            self,
            node: Node
    ) -> None:
        """Plot path found by A* so far for given `node`.
        """
        from matplotlib import pyplot as plt
        from matplotlib.cm import get_cmap
        from matplotlib import patches
        import pandas as pd

        p = self.retrace_path(node)
        cmap = get_cmap(plotcolormap)  # Dark2
        colors = cmap.colors
        fig, ax = plt.subplots(num=1, clear=True)

        # generate data points
        d = [
            (
                s.posx.expectation(),
                s.posy.expectation(),
                s.dirx.expectation(),
                s.diry.expectation(),
                f'{i}-Leaf#{s.leaf.idx if hasattr(s.leaf, "idx") else "ROOT"} '
                f'({s.posx.expectation():.2f},{s.posy.expectation():.2f}): '
                f'({s.dirx.expectation():.2f},{s.diry.expectation():.2f})'
            ) for i, s in enumerate(p)
        ]
        df = pd.DataFrame(data=d, columns=['X', 'Y', 'DX', 'DY', 'L'])

        # print direction arrows
        ax.quiver(
            df['X'],
            df['Y'],
            df['DX'],
            df['DY'],
            color=colors,
            width=0.001
        )

        # print goal position/area
        ax.add_patch(patches.Rectangle(
            (
                self.goalstate.posx.lower,
                self.goalstate.posy.lower
            ),
            self.goalstate.posx.upper - self.goalstate.posx.lower,
            self.goalstate.posy.upper - self.goalstate.posy.lower,
            linewidth=1,
            color='green',
            alpha=.2)
        )

        # annotate start and final position of agent as well as goal area
        ax.annotate('Start', (df['X'][0], df['Y'][0]))
        ax.annotate('End', (df['X'].iloc[-1], df['Y'].iloc[-1]))
        ax.annotate('Goal', (self.goalstate.posx.lower, self.goalstate.posy.lower))

        # scatter single steps
        for index, row in df.iterrows():
            ax.scatter(
                row['X'],
                row['Y'],
                marker='*',
                label=row['L'],
                color=colors[index]
            )

        # set figure/window/plot properties
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        fig.suptitle(str(node))
        plt.grid()
        plt.legend()
        plt.savefig(
            os.path.join(
                locs.logs,
                f'{os.path.basename(recent_example(os.path.join(locs.examples, "robotaction")))}-path.png'
            )
        )
        plt.show()


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
        n = Node(state=self.goalstate, g=0., h=0, parent=None)
        for l, tn, t in self.generate_steps(n):
            s = State(
                posx=Numeric().set(QuantileDistribution.from_cdf(l.distributions['x_in'].cdf.xshift(-l.distributions['x_out'].expectation()))),
                posy=Numeric().set(QuantileDistribution.from_cdf(l.distributions['y_in'].cdf.xshift(-l.distributions['y_out'].expectation()))),
                dirx=l.distributions['xdir_in'],  # in init, the candidate leaves can only be from MOVEFORWARD tree, which has no xdir_out distribution
                diry=l.distributions['ydir_in'],  # in init, the candidate leaves can only be from MOVEFORWARD tree, which has no ydir_out distribution
                ctree=t,
                leaf=l,
                tn=tn
            )
            # TODO: is the goal node the parent of the predecessors?
            n_ = Node(state=s, g=0., h=self.h(s), parent=None)
            if n_.h < self.goal_confidence:
                heapq.heappush(self.open, (n_.f, n_))

    def isgoal(
            self,
            node: Node
    ) -> bool:
        # true, if current belief state is sufficiently similar to init state and ancestor (i.e. first element in parent
        # chain) matches goal specification
        a = SubAStarBW.get_ancestor(node)
        return node.state.posx.p(self.initstate.posx.mpe()[1]) * node.state.posy.p(self.initstate.posy.mpe()[1]) >= self.state_similarity and \
            a.state.posx.p(self.goalstate.posx) * a.state.posy.p(self.goalstate.posy) >= self.goal_confidence

    def h(
            self,
            state: State
    ) -> float:
        # for backwards direction, the heuristic measures the probability, that the current state is the initstate
        p = 1
        if not any([x is None for x in [state.posx, state.posy]]):
            p *= state.posx.p(self.initstate.posx.mpe()[1]) * state.posy.p(self.initstate.posy.mpe()[1])
        # if not any([x is None for x in [state.dirx, state.diry]]):
        #     p *= state.dirx.p(self.initstate.dirx.mpe()[1]) * state.diry.p(self.initstate.diry.mpe()[1])
        return 1-p

    def stepcost(
            self,
            state
    ) -> float:
        # distance (Euclidean) travelled so far (from center of goal area to current position)
        cost = 0.
        gx = (self.goalstate.posx.upper - self.goalstate.posx.lower) / 2
        gy = (self.goalstate.posy.upper - self.goalstate.posy.lower) / 2

        dx = gx - state.posx.expectation()
        dy = gy - state.posy.expectation()
        cost += math.sqrt(dx ** 2 + dy ** 2)

        if 'angle' in state.leaf.value:
            cost += abs(state.leaf.value['angle'].expectation())

        return cost


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
                if v.name in query and v.name in l.distributions and v.name.replace('_in', '_out') in l.distributions:
                    ndist = Numeric().set(QuantileDistribution.from_cdf(l.distributions[v.name].cdf.xshift(-l.distributions[v.name.replace('_in', '_out')].expectation())))
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
        if isinstance(node.state, Goal):
            query = {
                'x_in': node.state.posx,
                'y_in': node.state.posy
            }
        else:
            query = {}
            if 'x_in' in node.state.leaf.path:
                query['x_in'] = node.state.leaf.distributions['x_in'].mpe()[1]

            if 'y_in' in node.state.leaf.path:
                query['y_in'] = node.state.leaf.distributions['y_in'].mpe()[1]

            if 'xdir_in' in node.state.leaf.path:
                query['xdir_in'] = node.state.leaf.distributions['xdir_in'].mpe()[1]

            if 'ydir_in' in node.state.leaf.path:
                query['ydir_in'] = node.state.leaf.distributions['ydir_in'].mpe()[1]

        return [
            (leaf, treename, tree) for treename, tree in self.models.items() for _, leaf in self.reverse(
                t=tree,
                query=tree.bind(
                    {
                        k: v for k, v in query.items() if k in tree.varnames
                    },
                    allow_singular_values=False
                ),
                confidence=.0  # self.goal_confidence?
            )
        ]

    def generate_successors(
            self,
            node: Node
    ) -> List[Node]:

        predecessors = []
        for pred, tn, t in self.generate_steps(node):

            # get distributions representing current belief state
            posx = node.state.posx
            posy = node.state.posy
            dirx = node.state.dirx
            diry = node.state.diry

            # update new position distributions to reflect the actual (absolute) position after execution
            if 'x_in' in pred.distributions and 'x_out' in pred.distributions:
                posx = Numeric().set(QuantileDistribution.from_cdf(pred.distributions['x_in'].cdf.xshift(-pred.distributions['x_out'].expectation())))

            if 'y_in' in pred.distributions and 'y_out' in pred.distributions:
                posy = Numeric().set(QuantileDistribution.from_cdf(pred.distributions['y_in'].cdf.xshift(-pred.distributions['y_out'].expectation())))

            # update new orientation distributions to reflect the actual (absolute) direction after execution
            if 'xdir_in' in pred.distributions and 'xdir_out' in pred.distributions:
                dirx = Numeric().set(QuantileDistribution.from_cdf(pred.distributions['xdir_in'].cdf.xshift(-pred.distributions['xdir_out'].expectation())))

            if 'ydir_in' in pred.distributions and 'ydir_out' in pred.distributions:
                diry = Numeric().set(QuantileDistribution.from_cdf(pred.distributions['ydir_in'].cdf.xshift(-pred.distributions['ydir_out'].expectation())))

            # initialize new belief state for potential predecessor
            state = State(
                posx=posx,
                posy=posy,
                ctree=t,
                leaf=pred,
                dirx=dirx,
                diry=diry,
                tn=tn
            )

            predecessors.append(
                Node(
                    state=state,
                    g=node.g + self.stepcost(state),
                    h=self.h(state),
                    parent=node
                )
            )

        return predecessors


if __name__ == "__main__":
    init_loggers(level='debug')
    recent = recent_example(os.path.join(locs.examples, 'robotaction'))

    logger.debug(f'Loading trees from {recent}...')
    models = dict(
        [
            (
                treefile.name,
                JPT.load(str(treefile))
            )
            for p in [recent]
            for treefile in Path(p).rglob('*.tree')
        ]
    )

    logger.debug('...done! Plotting initial distribution...')

    jpt_ = models['MOVEFORWARD.tree']
    # Move.plot(
    #     jpt_=jpt_,
    #     qvarx=jpt_.varnames['x_out'],
    #     qvary=jpt_.varnames['y_out'],
    #     evidence=None,
    #     title=r'Init',
    #     # conf=.0003,
    #     limx=(-100, 100),
    #     limy=(-100, 100),
    #     # limz=(0, 0.001),
    #     # save=os.path.join(locs.logs, f'{datetime.datetime.now().strftime(FILESTRFMT_SEC)}.png'),
    #     show=True
    # )

    logger.debug('...done! Initializing start and goal states...')

    tolerance = .1

    initx, inity, initdirx, initdiry = [-75, 75, 0, -1]
    posx = ContinuousSet(initx - abs(tolerance * initx), initx + abs(tolerance * initx))
    posy = ContinuousSet(inity - abs(tolerance * inity), inity + abs(tolerance * inity))
    dirx = ContinuousSet(initdirx - abs(tolerance * initdirx), initdirx + abs(tolerance * initdirx))
    diry = ContinuousSet(initdiry - abs(tolerance * initdiry), initdiry + abs(tolerance * initdiry))

    posteriors = models['MOVEFORWARD.tree'].posterior(
        evidence={
            'x_in': posx,
            'y_in': posy,
            'xdir_in': dirx,
            'ydir_in': diry
        }
    )

    initstate = State(
        posx=posteriors['x_in'],
        posy=posteriors['y_in'],
        dirx=posteriors['xdir_in'],
        diry=posteriors['ydir_in'],
    )

    # initstate.plot(show=True)

    goalx, goaly = [-75, 66]
    goalstate = Goal(
        posx=ContinuousSet(goalx - abs(tolerance * goalx), goalx + abs(tolerance * goalx)),
        posy=ContinuousSet(goaly - abs(tolerance * goaly), goaly + abs(tolerance * goaly))
    )

    logger.debug('...done! Initializing A* Algorithm...')

    # a_star = SubAStar(
    #     initstate=initstate,
    #     goalstate=goalstate,
    #     models=models
    # )

    a_star = SubAStarBW(
        initstate=initstate,
        goalstate=goalstate,
        models=models
    )

    # a_star = BiDirAStar(
    #     SubAStar,
    #     SubAStar_BW,
    #     initstate,
    #     goalstate,
    #     state_similarity=.9,
    #     goal_confidence=.01,
    #     models=models
    # )
    logger.debug('...done! Starting search...')

    path = a_star.search()
    logger.debug('...done!', path)
