import os
from typing import Tuple, Dict, Any

import numpy as np
from dnutils import ifnone, first
from matplotlib import pyplot as plt

from calo.core.astar_jpt import State, SubAStar, Goal, SubAStarBW
from calo.utils import locs
from calo.utils.constants import plotcolormap
from calo.utils.utils import recent_example
from jpt.base.intervals import ContinuousSet
from jpt.trees import Node


class State_(State):

    def __init__(self):
        super().__init__()

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
        x = self['x_out'].pdf.boundaries()
        y = self['y_out'].pdf.boundaries()

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


class SubAStar_(SubAStar):

    def __init__(
            self,
            initstate: State_,
            goal: Goal,
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

    def stepcost(
            self,
            state
    ) -> float:
            # distance travelled and/or angle turnt in current step
            # cost = 0.
            #
            # if 'x_out' in self.models[state['tree']].leaves[state['leaf']].distributions:
            #     cost += abs(self.models[state['tree']].leaves[state['leaf']].distributions['x_out'].expectation())
            #
            # if 'y_out' in self.models[state['tree']].leaves[state['leaf']].distributions:
            #     cost += abs(self.models[state['tree']].leaves[state['leaf']].distributions['y_out'].expectation())
            #
            # if 'angle' in self.models[state['tree']].leaves[state['leaf']].distributions:
            #     cost += abs(self.models[state['tree']].leaves[state['leaf']].distributions['angle'].expectation())

            return 1

    def h(
            self,
            state: State_
    ) -> float:
        # p = 1
        # if 'x_out' in state and 'y_out' in state:
        #     p *= state['x_out'].p(self.goal['x_out']) * state['y_out'].p(self.goal['y_out'])
        #
        # return 1 - p

        # manhattan distance
        c = 0.
        if 'x_in' in state:
            if isinstance(state['x_in'], set):
                vx = min(list(state['x_in']))
            else:
                vx = min(list(state['x_in'].mpe()[1]))
            c += abs(vx - first(self.goal['x_in']))
        if 'y_in' in state:
            if isinstance(state['y_in'], set):
                vy = min(list(state['y_in']))
            else:
                vy = min(list(state['y_in'].mpe()[1]))
            c += abs(vy - first(self.goal['y_in']))
        return c

    def plot(
            self,
            node: Node
    ) -> None:
        """ONLY FOR GRIDWORLD DATA
        """
        from matplotlib import pyplot as plt
        from matplotlib import colormaps
        from matplotlib import patches
        import pandas as pd

        p = self.retrace_path(node)
        cmap = colormaps[plotcolormap]
        colors = cmap.colors
        fig, ax = plt.subplots(num=1, clear=True)

        # generate data points
        d = [
            (
                s['x_in'].expectation(),
                s['y_in'].expectation(),
                s['xdir_in'].expectation(),
                s['ydir_in'].expectation(),
                f'{i}-Leaf#{s.leaf if s.leaf is not None else "ROOT"} '
                f'({s["x_in"].expectation():.2f},{s["y_in"].expectation():.2f}): '
                f'({s["xdir_in"].expectation():.2f},{s["ydir_in"].expectation():.2f})'
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

        gxl = self.goal['x_in'].lower if isinstance(self.goal['x_in'], ContinuousSet) else first(self.goal['x_in'])
        gxu = self.goal['x_in'].upper if isinstance(self.goal['x_in'], ContinuousSet) else first(self.goal['x_in'])
        gyl = self.goal['y_in'].lower if isinstance(self.goal['y_in'], ContinuousSet) else first(self.goal['y_in'])
        gyu = self.goal['y_in'].upper if isinstance(self.goal['y_in'], ContinuousSet) else first(self.goal['y_in'])

        # print goal position/area
        ax.add_patch(patches.Rectangle(
            (
                gxl,
                gyl
            ),
            gxu - gxl,
            gyu - gyl,
            linewidth=1,
            color='green',
            alpha=.2)
        )

        # annotate start and final position of agent as well as goal area
        ax.annotate('Start', (df['X'][0], df['Y'][0]))
        ax.annotate('End', (df['X'].iloc[-1], df['Y'].iloc[-1]))
        ax.annotate('Goal', (gxl, gyl))

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

    # def plot(
    #         self,
    #         node: Node
    # ) -> None:
    #     """Plot path found by A* so far for given `node`.
    #     """
    #     from matplotlib import pyplot as plt
    #     from matplotlib.cm import get_cmap
    #     from matplotlib import patches
    #     import pandas as pd
    #
    #     p = self.retrace_path(node)
    #     cmap = get_cmap(plotcolormap)  # Dark2
    #     colors = cmap.colors
    #     fig, ax = plt.subplots(num=1, clear=True)
    #
    #     # generate data points
    #     d = [
    #         (
    #             s.posx.expectation(),
    #             s.posy.expectation(),
    #             s.dirx.expectation(),
    #             s.diry.expectation(),
    #             f'{i}-Leaf#{s.leaf.idx if hasattr(s.leaf, "idx") else "ROOT"} '
    #             f'({s.posx.expectation():.2f},{s.posy.expectation():.2f}): '
    #             f'({s.dirx.expectation():.2f},{s.diry.expectation():.2f})'
    #         ) for i, s in enumerate(p)
    #     ]
    #     df = pd.DataFrame(data=d, columns=['X', 'Y', 'DX', 'DY', 'L'])
    #
    #     # print direction arrows
    #     ax.quiver(
    #         df['X'],
    #         df['Y'],
    #         df['DX'],
    #         df['DY'],
    #         color=colors,
    #         width=0.001
    #     )
    #
    #     # print goal position/area
    #     ax.add_patch(patches.Rectangle(
    #         (
    #             self.goal.posx.lower,
    #             self.goal.posy.lower
    #         ),
    #         self.goal.posx.upper - self.goal.posx.lower,
    #         self.goal.posy.upper - self.goal.posy.lower,
    #         linewidth=1,
    #         color='green',
    #         alpha=.2)
    #     )
    #
    #     # annotate start and final position of agent as well as goal area
    #     ax.annotate('Start', (df['X'][0], df['Y'][0]))
    #     ax.annotate('End', (df['X'].iloc[-1], df['Y'].iloc[-1]))
    #     ax.annotate('Goal', (self.goal.posx.lower, self.goal.posy.lower))
    #
    #     # scatter single steps
    #     for index, row in df.iterrows():
    #         ax.scatter(
    #             row['X'],
    #             row['Y'],
    #             marker='*',
    #             label=row['L'],
    #             color=colors[index]
    #         )
    #
    #     # set figure/window/plot properties
    #     ax.set_xlabel(r'$x$')
    #     ax.set_ylabel(r'$y$')
    #     fig.suptitle(str(node))
    #     plt.grid()
    #     plt.legend()
    #     plt.savefig(
    #         os.path.join(
    #             locs.logs,
    #             f'{os.path.basename(recent_example(os.path.join(locs.examples, "robotaction")))}-path.png'
    #         )
    #     )
    #     plt.show()


class SubAStarBW_(SubAStarBW):

    def __init__(
            self,
            initstate: State_,  # would be the goal state of forward-search
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

    def stepcost(
            self,
            state
    ) -> float:
        # distance (Manhattan) travelled/turnt in current step
        # cost = 0.

        # if 'x_out' in self.models[state['tree']].leaves[state['leaf']].distributions:
        #     cost += abs(self.models[state['tree']].leaves[state['leaf']].distributions['x_out'].expectation())
        #
        # if 'y_out' in self.models[state['tree']].leaves[state['leaf']].distributions:
        #     cost += abs(self.models[state['tree']].leaves[state['leaf']].distributions['y_out'].expectation())
        #
        # if 'angle' in self.models[state['tree']].leaves[state['leaf']].distributions:
        #     cost += max(1, abs(self.models[state['tree']].leaves[state['leaf']].distributions['angle'].expectation()))

        return 1

    def h(
            self,
            state: State_
    ) -> float:
        # for backwards direction, the heuristic measures the probability, that the current state is the initstate
        # p = 1
        # FIXME:
        # if 'x_in' in state and 'y_in' in state:
        #     p *= self.initstate['x_in'].p(state['x_in']) * self.initstate['y_in'].p(state['x_in'])
        #
        # # ... and faces the initial direction specified
        # if 'xdir_in' in state and 'ydir_in' in state:
        #     p *= self.initstate['xdir_in'].p(state['xdir_in']) * self.initstate['ydir_in'].p(state['xdir_in'])
        # return 1 - p

        # manhattan distance
        c = 0.
        if 'x_in' in state:
            if isinstance(state['x_in'], set):
                vx = min(list(state['x_in']))
            else:
                vx = min(list(state['x_in'].mpe()[1]))
            c += abs(vx - self.initstate['x_in'].expectation())
        if 'y_in' in state:
            if isinstance(state['y_in'], set):
                vy = min(list(state['y_in']))
            else:
                vy = min(list(state['y_in'].mpe()[1]))
            c += abs(vy - self.initstate['y_in'].expectation())
        return c

    # def retrace_path(
    #         self,
    #         node
    # ) -> Any:
    #     return list(reversed(super().retrace_path(node)))

    def plot(
            self,
            node
    ) -> None:
        pass