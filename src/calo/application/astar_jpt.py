import datetime
import math
import math
import os
from pathlib import Path
from typing import List, Dict, Union, Any

import dnutils
import pyximport
from calo.core.astar import AStar, Node
from calo.logs.logs import init_loggers
from calo.models.action import Move
from calo.utils import locs
from calo.utils.constants import calologger, plotcolormap, FILESTRFMT_SEC
from calo.utils.utils import pnt2line_alt, recent_example, angledeg
from jpt.trees import JPT

pyximport.install()
from jpt.base.intervals import ContinuousSet


logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


class State:
    def __init__(
            self,
            posx: Union[float, ContinuousSet],
            posy: Union[float, ContinuousSet],
            dirx: Union[float, ContinuousSet] = None,
            diry: Union[float, ContinuousSet] = None,
            ctree: Any = None,
            leaf: Any = None
    ):
        self.posx = posx
        self.posy = posy
        self.dirx = dirx
        self.diry = diry
        self.ctree = ctree
        self.leaf = leaf

    def __eq__(self, other):
        return self.posx == other.posx and self.posy == other.posy

    def __str__(self):
        return f'State<pos: ({str(self.posx) if isinstance(self.posx, ContinuousSet) else round(self.posx, 2)}/' \
               f'{str(self.posy) if isinstance(self.posy, ContinuousSet) else round(self.posy) });' \
               f'{f" dir: ({str(self.dirx)}/{str(self.diry)})" if isinstance(self.dirx, ContinuousSet) and isinstance(self.diry, ContinuousSet) else f" dir: ({round(self.dirx, 2)}/{round(self.diry, 2)})" if self.dirx is not None and self.diry is not None else ""}>'

    def __repr__(self):
        return str(self)


class SubAStar(AStar):

    def __init__(
            self,
            initstate: State,
            goalstate: State,  # might be belief state later
            models: Dict
    ):
        self.models = models
        super().__init__(initstate, goalstate)

    def stepcost(
            self,
            state
    ) -> float:
        # distance (Euclidean) travelled so far (from init_pos to current position)
        cost = 0.
        dx = self.initstate.posx - state.posx
        dy = self.initstate.posy - state.posy
        cost += math.sqrt(dx ** 2 + dy ** 2)

        # difference in orientation (from init dir to current dir)
        # cost += angledeg([state.dirx, state.diry], [self.initstate.dirx, self.initstate.diry])
        if 'angle' in state.leaf.value:
            cost += abs(state.leaf.value['angle'].expectation())

        return cost

    def h(
            self,
            state
    ) -> float:
        # Euclidean distance from current position to goal node
        cost = 0.
        gx = self.goalstate.posx
        gy = self.goalstate.posy
        if isinstance(self.goalstate.posx, ContinuousSet) and isinstance(self.goalstate.posy, ContinuousSet):
            # assuming the goal area is a rectangle, calculate the minimum distance between the current position (= point)
            # to the nearest edge of the rectangle
            xl = self.goalstate.posx.lower
            xu = self.goalstate.posx.upper
            yl = self.goalstate.posy.lower
            yu = self.goalstate.posy.upper
            gx = xl + (xu-xl)/2
            gy = yl + (yu-yl)/2

            cost += min([d for d, _ in [
                pnt2line_alt([state.posx, state.posy], [xl, yl], [xl, yu]),
                pnt2line_alt([state.posx, state.posy], [xl, yl], [xl, yu]),
                pnt2line_alt([state.posx, state.posy], [xl, yu], [xu, yu]),
                pnt2line_alt([state.posx, state.posy], [xu, yl], [xu, yu])
            ]])
        else:
            # current position and goal position are points
            dx = state.posx - self.goalstate.posx
            dy = state.posy - self.goalstate.posy
            cost += math.sqrt(dx ** 2 + dy ** 2)

        # if no directions are given, return costs at this point
        if any([x is None for x in [state.dirx, state.diry]]):
            return cost  # TODO

        # difference in orientation (current dir to dir to goal node)
        # vec to goal node:
        dx = gx - state.posx
        dy = gy - state.posy
        cost += angledeg([state.dirx, state.diry], [dx, dy])

        return cost

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
            'x_in': node.state.posx,
            'y_in': node.state.posy,
            'xdir_in': node.state.dirx,
            'ydir_in': node.state.diry
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
        #     jpt_.plot(
        #         title=i,
        #         plotvars=jpt_.variables,
        #         leaffill='#CCDAFF',
        #         nodefill='#768ABE',
        #         alphabet=True
        #         )
        #         if i == 'MOVEFORWARD.tree' and jpt_:
        #             Move.plot(
        #                 jpt_=jpt_,
        #                 qvarx=jpt_.varnames['x_out'],
        #                 qvary=jpt_.varnames['y_out'],
        #                 evidence={jpt_.varnames[k]: v for k, v in evidence.items()},
        #                 title=f'{i} (conditional)',
        #                 # conf=.0003,
        #                 limx=(-150, 150),
        #                 limy=(-150, 150),
        #                 # limz=(0, 0.001),
        #                 save=os.path.join(locs.logs, f'{i}_cond-{datetime.datetime.now().strftime(FILESTRFMT_SEC)}.png'),
        #                 show=False,
        #                 alphabet=True
        #             )

        return [(leaf, treename, tree) for treename, tree in condtrees if tree is not None for _, leaf in tree.leaves.items()]

    def generate_successors(
            self,
            node
    ) -> List[Node]:
        successors = []
        for succ, tn, t in self.generate_steps(node):

            # update position
            pos_x = node.state.posx
            pos_y = node.state.posy
            if 'x_out' in succ.value and 'y_out' in succ.value:
                pos_x += succ.value['x_out'].expectation()
                pos_y += succ.value['y_out'].expectation()

            # update direction
            dir_x = node.state.dirx
            dir_y = node.state.diry
            if 'xdir_out' in succ.value and 'ydir_out' in succ.value:
                dir_x += succ.value['xdir_out'].expectation()
                dir_y += succ.value['ydir_out'].expectation()

            state = State(
                posx=pos_x,
                posy=pos_y,
                ctree=t,
                leaf=succ,
                dirx=dir_x,
                diry=dir_y
            )

            # if new candidate doesn't add anything to current state
            if state.posx == node.state.posx and \
                state.posy == node.state.posy and \
                state.dirx == node.state.dirx and \
                state.diry == node.state.diry:
                continue

            # TODO: check model collision? --> succ.value['collision'].expectation()

            successors.append(
                Node(
                    state=state,
                    g=node.g + self.stepcost(state),
                    h=self.h(state),
                    parent=node,
                    tree=tn,  # TODO: remove! --> only for debugging
                    leaf=succ  # TODO: remove! --> only for debugging
                )
            )
        return successors

    def isgoal(
            self,
            node
    ) -> bool:
        # true if node pos and target pos are equal
        if isinstance(self.goalstate.posx, ContinuousSet) and isinstance(self.goalstate.posy, ContinuousSet):
            return self.goalstate.posx.contains_value(node.state.posx) and self.goalstate.posy.contains_value(node.state.posy)
        else:
            return node.state == self.goalstate

    def plot(
            self,
            node: Node
    ) -> None:
        '''Print path found by A* so far for given `node`.'''
        from matplotlib import pyplot as plt
        from matplotlib.cm import get_cmap
        from matplotlib import patches
        import pandas as pd

        # retrace
        p = self.retrace_path(node)
        cmap = get_cmap(plotcolormap)  # Dark2
        colors = cmap.colors
        fig, ax = plt.subplots(num=1, clear=True)

        d = [
            (
                s.posx,
                s.posy,
                s.dirx,
                s.diry,
                f'{i}-Leaf#{s.leaf.idx if hasattr(s.leaf, "idx") else "ROOT"} ({round(s.posx, 2)},{round(s.posy, 2)}): ({round(s.dirx, 2)},{round(s.diry, 2)})'
            ) for i, s in enumerate(p)
        ]
        df = pd.DataFrame(data=d, columns=['X', 'Y', 'DX', 'DY', 'L'])

        # print direction arrows
        ax.quiver(
            df['X'],
            df['Y'],
            df['DX'],
            df['DY'],
            color='k',
            width=0.002
        )

        # annotate start and final position of agent
        ax.annotate('START', (df['X'][0], df['Y'][0]))
        ax.annotate('FINAL', (df['X'].iloc[-1], df['Y'].iloc[-1]))

        # print goal position/area
        if isinstance(self.goalstate.posx, ContinuousSet) and isinstance(self.goalstate.posy, ContinuousSet):
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
            ax.annotate('GOAL', (self.goalstate.posx.lower, self.goalstate.posy.lower))
        else:
            ax.scatter(
                self.goalstate.posx,
                self.goalstate.posy,
                marker='*',
                c=colors[-1]
            )
            ax.annotate('GOAL', (self.goalstate.posx, self.goalstate.posy))

        # scatter single steps
        for index, row in df.iterrows():
            ax.scatter(
                row['X'],
                row['Y'],
                marker='*',
                label=row['L'],
                c=colors[index]
            )

        fig.suptitle(str(node))
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(locs.logs, f'{os.path.basename(recent_example(os.path.join(locs.examples, "robotaction")))}-path.png'))
        plt.show()


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

    initx, inity, initdirx, initdiry = [-75, 75, 1, 0]
    # initx, inity, initdirx, initdiry = [-72.0337547702897, -71.8206700251736, -0.414975684428278,	0.909832501800899]
    # initx, inity, initdirx, initdiry = [.0, .0, -1., .0]

    initstate = State(
        posx=initx,
        posy=inity,
        dirx=initdirx,
        diry=initdiry,
    )
    # initstate = State(
    #     posx=ContinuousSet(initx, initx),
    #     posy=ContinuousSet(inity, inity),
    #     dirx=ContinuousSet(initdirx, initdirx),
    #     diry=ContinuousSet(initdiry, initdiry)
    # )

    goalx, goaly = [-75, 66]
    # goalx, goaly = [-75.301941580268, -58.9992353237625]
    # goalx, goaly = [-76.2808983901234, -51.5095777048669]  # [4.5, -2.5]

    goalstate = State(
        posx=ContinuousSet(goalx - abs(tolerance * goalx), goalx + abs(tolerance * goalx)),
        posy=ContinuousSet(goaly - abs(tolerance * goaly), goaly + abs(tolerance * goaly))
    )

    logger.debug('...done! Initializing A* Algorithm...')

    a_star = SubAStar(
        initstate=initstate,
        goalstate=goalstate,
        models=models
    )

    logger.debug('...done! Starting search...')

    path = a_star.search()
    logger.debug('...done!', path)
