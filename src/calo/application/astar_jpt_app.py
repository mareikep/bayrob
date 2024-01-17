import os
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dnutils import ifnone, first
from jpt.base.intervals import ContinuousSet
from matplotlib import pyplot as plt
from plotly.graph_objs import Figure

from calo.core.astar_jpt import State, SubAStar, Goal, SubAStarBW
from calo.utils import locs
from calo.utils.plotlib import plot_heatmap, plot_scatter_quiver, defaultconfig, plotly_pt, plotly_sq, \
    plot_dists_layered, fig_to_file
from calo.utils.utils import fmt
from jpt.trees import Node

#
# class State_(State):
#
#     def __init__(self, d: dict=None):
#         super().__init__(d=d)
#
#     def plot(
#             self,
#             title: str = None,
#             conf: float = None,
#             limx: Tuple = None,
#             limy: Tuple = None,
#             limz: Tuple = None,
#             save: str = None,
#             show: bool = True
#     ) -> None:
#         """Plots a heatmap representing the belief state for the agents' position, i.e. the joint
#         probability of the x and y variables: P(x, y).
#
#         :param title: The plot title
#         :param conf:  A confidence value. Values below this threshold are set to 0. (= equal color for lowest value in plot)
#         :param limx: The limits for the x-variable; determined from boundaries if not given
#         :param limy: The limits for the y-variable; determined from boundaries if not given
#         :param limz: The limits for the z-variable; determined from data if not given
#         :param save: The location where the plot is saved (if given)
#         :param show: Whether the plot is shown
#         :return: None
#         """
#         # generate datapoints
#         x = self['x_in'].pdf.boundaries()
#         y = self['y_in'].pdf.boundaries()
#
#         # determine limits
#         xmin = ifnone(limx, min(x) - 15, lambda l: l[0])
#         xmax = ifnone(limx, max(x) + 15, lambda l: l[1])
#         ymin = ifnone(limy, min(y) - 15, lambda l: l[0])
#         ymax = ifnone(limy, max(y) + 15, lambda l: l[1])
#
#         X, Y = np.meshgrid(x, y)
#         Z = np.array(
#             [
#                 self['x_in'].pdf(x) * self['y_in'].pdf(y)
#                 for x, y, in zip(X.ravel(), Y.ravel())
#             ]).reshape(X.shape)
#
#         # show only values above a certain threshold, consider lower values as high-uncertainty areas
#         if conf is not None:
#             Z[Z < conf] = 0.
#
#         # remove or replace by eliminating values > median
#         Z[Z > np.median(Z)] = np.median(Z)
#
#         zmin = ifnone(limz, Z.min(), lambda l: l[0])
#         zmax = ifnone(limz, Z.max(), lambda l: l[1])
#
#         # init plot
#         fig, ax = plt.subplots(num=1, clear=True)
#         fig.patch.set_facecolor('#D6E7F8')  # set bg color around the plot area (royal purple)
#         ax.set_facecolor('white')  # set bg color of plot area (dark purple)
#         cmap = 'BuPu'  # viridis, Blues, PuBu, 0rRd, BuPu
#
#         # generate heatmap
#         c = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=zmin, vmax=zmax)
#         ax.set_title(f'P(x,y)')
#
#         # setting the limits of the plot to the limits of the data
#         ax.axis([xmin, xmax, ymin, ymax])
#         # ax.axis([-100, 100, -100, 100])
#         ax.set_xlabel(r'$x$')
#         ax.set_ylabel(r'$y$')
#         fig.colorbar(c, ax=ax)
#         fig.suptitle(title)
#         fig.canvas.manager.set_window_title(f'Belief State: P(x/y)')
#
#         if save:
#             plt.savefig(save)
#
#         if show:
#             plt.show()


class SubAStar_(SubAStar):

    def __init__(
            self,
            initstate: State,
            goal: Goal,
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

    def stepcost(
            self,
            state,
            parent
    ) -> float:

        return self.dist(state, parent)

    def h(
            self,
            state: State
    ) -> float:
        # for forward direction, the heuristic measures the mean of the distances of the current state's variables
        # and the ones from the goal state. If `state` does not contain all variables of the goalstate (which typically
        # applies to the goal state), the distance is infinite.
        if not set(self.goal.keys()).issubset(set(state.keys())): return np.inf

        return self.dist(state, self.goal)

    def plot_pos(
            self,
            path: List,
            title: str = None,
            conf: float = None,
            limx: Tuple = None,
            limy: Tuple = None,
            limz: Tuple = None,
            save: str = None,
            show: bool = True
    ) -> Figure:

        # generate datapoints
        data = pd.DataFrame(
                data=[
                    self.gendata(
                        'x_in',
                        'y_in',
                        s,
                        conf=conf
                    ) for s in path
                ],
                columns=['x_in', 'y_in', 'z', 'lbl']
            )

        if not data.empty:
            return plot_heatmap(
                xvar='x_in',
                yvar='y_in',
                data=data,
                title=title,
                limx=limx,
                limy=limy,
                limz=limz,
                save=save,
                show=show
            )

    def plot_dir(
            self,
            path: List,
            title: str = None,
            conf: float = None,
            limx: Tuple = None,
            limy: Tuple = None,
            limz: Tuple = None,
            save: str = None,
            show: bool = True
    ) -> Figure:

        # generate datapoints
        data = pd.DataFrame(
                data=[
                    self.gendata(
                        'xdir_in',
                        'ydir_in',
                        s,
                        conf=conf
                    ) for s in path
                ],
                columns=['xdir_in', 'ydir_in', 'z', 'lbl']
            )

        if not data.empty:
            return plot_heatmap(
                xvar='xdir_in',
                yvar='ydir_in',
                data=data,
                title=title,
                limx=limx,
                limy=limy,
                limz=limz,
                save=save,
                show=show
            )

    def plot_path(
            self,
            xvar,
            yvar,
            p: List,
            obstacles: List = None,
            title: str = None,
            save: str = None,
            show: bool = False,
            limx: Tuple = None,
            limy: Tuple = None
    ) -> Figure:

        # generate data points
        d = [
            (
                np.mean([s[xvar].mpe()[0].lower, s[xvar].mpe()[0].upper]),          # x
                np.mean([s[yvar].mpe()[0].lower, s[yvar].mpe()[0].upper]),          # y
                np.mean([s['xdir_in'].mpe()[0].lower, s['xdir_in'].mpe()[0].upper]),     # dx
                np.mean([s['ydir_in'].mpe()[0].lower, s['ydir_in'].mpe()[0].upper]),     # dy
                f'Step {i}',                    # step
                f'<b>Step {i}: {"root" if s.leaf is None or s.tree is None else f"{s.tree}-Leaf#{s.leaf}"}</b><br><b>MPEs:</b><br>{"<br>".join(f"{k}: {fmt(v)}" for k, v in s.items())}',  # lbl
                1                               # size
            )
            for i, s in enumerate(p)
        ]

        # generate data for scatter circles and quivers
        data = pd.DataFrame(
            data=d,
            columns=[xvar, yvar, 'dx', 'dy', 'step', 'lbl', 'size']
        )

        # generate data for distribution blobs
        data_dists = pd.DataFrame(
                data=[
                    self.gendata(
                        xvar,
                        yvar,
                        s,
                    ) for s in p
                ],
                columns=[xvar, yvar, 'z', 'lbl']
            )

        # generate data for init dist blob
        data_init = pd.DataFrame(
            data=[
                self.gendata(
                    xvar,
                    yvar,
                    self.initstate
                )
            ],
            columns=[xvar, yvar, 'z', 'lbl']
        )

        # determine corners of goal area
        gxl = self.goal[xvar].lower if isinstance(self.goal[xvar], ContinuousSet) else first(self.goal[xvar])
        gxu = self.goal[xvar].upper if isinstance(self.goal[xvar], ContinuousSet) else first(self.goal[xvar])
        gyl = self.goal[yvar].lower if isinstance(self.goal[yvar], ContinuousSet) else first(self.goal[yvar])
        gyu = self.goal[yvar].upper if isinstance(self.goal[yvar], ContinuousSet) else first(self.goal[yvar])

        # determine position and direction of initstate
        ix = np.mean([self.initstate[xvar].mpe()[0].lower, self.initstate[xvar].mpe()[0].upper])
        iy = np.mean([self.initstate[yvar].mpe()[0].lower, self.initstate[yvar].mpe()[0].upper])
        ixd = np.mean([self.initstate['xdir_in'].mpe()[0].lower, self.initstate['xdir_in'].mpe()[0].upper])
        iyd = np.mean([self.initstate['ydir_in'].mpe()[0].lower, self.initstate['ydir_in'].mpe()[0].upper])

        mainfig = go.Figure()

        # plot obstacles in background
        if obstacles is not None:
            for (o, on) in obstacles:
                mainfig.add_trace(
                    plotly_sq(o, lbl=on, color='rgb(15,21,110)', legend=False))

        fig_initstate = go.Figure()

        # generate heatmap plots for init distribution blob
        fig_path_init = plot_dists_layered(
            xvar,
            yvar,
            data_init,
            limx=limx,
            limy=limy,
            show=False
        )

        # add init distribution blob to main plot
        fig_initstate.add_traces(
            data=fig_path_init.data
        )

        # generate initstate plot
        fig_initstate.add_traces(
            data=plotly_pt(
                pt=(ix, iy),
                dir=(ixd, iyd),
                name=f"Start<br>x_in: {ix}<br>y_in: {iy}",
                color='rgb(0,127,0)'
            ).data
        )

        # generate goal area plot
        fig_initstate.add_trace(
            plotly_sq(
                area=(gxl, gyl, gxu, gyu),
                lbl=f"Goal Area",
                legend=False,
                color='rgb(0,127,0)'
            )
        )

        # add initstate and goal area plots to main plot
        mainfig.add_traces(
            data=fig_initstate.data
        )

        # generate heatmap plots for distribution blobs
        fig_path_dists = plot_dists_layered(
            xvar,
            yvar,
            data_dists,
            limx=limx,
            limy=limy,
            show=False
        )

        # add distribution blobs to main plot
        mainfig.add_traces(
            data=fig_path_dists.data
        )

        # generate scatter/quiver plot for steps
        fig_path = plot_scatter_quiver(
            xvar,
            yvar,
            data,
            show=False
        )

        # add scatter/quiver plot to main plot
        mainfig.add_traces(
            data=fig_path.data
        )

        # set range and size of main plot
        mainfig.update_layout(
            xaxis=dict(
                title=xvar,
                side='top',
                range=[*limx]
            ),
            yaxis=dict(
                title=yvar,
                range=[*limy]
            ),
            height=1000,
            width=1100,
            title=title
        )

        if save:
            fig_to_file(mainfig, save)

        if show:
            mainfig.show(config=defaultconfig(save))

        return mainfig

    def gendata(
            self,
            xvar,
            yvar,
            state,
            conf: float = None
    ):

        # generate datapoints
        x = state[xvar].pdf.boundaries()
        y = state[yvar].pdf.boundaries()

        X, Y = np.meshgrid(x, y)
        Z = np.array(
            [
                state[xvar].pdf(x) * state[yvar].pdf(y)
                for x, y, in zip(X.ravel(), Y.ravel())
            ]).reshape(X.shape)

        # show only values above a certain threshold, consider lower values as high-uncertainty areas
        if conf is not None:
            Z[Z < conf] = 0.

        # remove or replace by eliminating values > median
        Z[Z > np.median(Z)] = np.median(Z)

        lbl = (f'<b>{"ROOT" if state.leaf is None or state.tree is None else f"{state.tree}-Leaf#{state.leaf}"}</b><br>'
               f'<b>MPEs:</b><br>'
               f'{"<br>".join(f"{k}: {fmt(v)}" for k, v in state.items())}<br>'
               f'<b>Expectations:</b><br>'
               f'{"<br>".join(f"{k}: {fmt(v.expectation())}" for k, v in state.items())}<br>')  # lbl

        params = f"<b>Params (MPEs):</b><br>"
        params += "None" if state.tree is None or state.leaf is None else f'{",<br>".join([f"<i>{v.name}:</i> {fmt(self.models.get(state.tree).leaves[state.leaf].distributions[v])}" for v in self.models.get(state.tree).features if v not in state])}'

        return x, y, Z, lbl+params

    def plot(
            self,
            node: Node,
            **kwargs
    ) -> Figure:
        """ONLY FOR GRIDWORLD DATA
        """

        p = self.retrace_path(node)

        # plot annotated rectangles representing the obstacles and world boundaries
        obstacles = [
            ((0, 0, 100, 100), "kitchen_boundaries"),
            ((15, 10, 25, 20), "chair1"),
            ((35, 10, 45, 20), "chair2"),
            ((10, 30, 50, 50), "kitchen_island"),
            ((80, 30, 100, 70), "stove"),
            ((10, 80, 50, 100), "kitchen_unit"),
            ((60, 80, 80, 100), "fridge"),
        ]

        fig=self.plot_path(
            xvar='x_in',
            yvar='y_in',
            p=p,
            obstacles=obstacles,
            title=f'SubAStar (fwd)<br>{str(node)}',
            save=os.path.join(locs.logs, f'{os.path.basename(os.path.join(locs.logs, "ASTAR-fwd-path.html"))}'),
            show=True,
            limx=(0, 100),
            limy=(0, 100)
        )

        if kwargs.get('plotdistributions', True):
            # plot distributions
            self.plot_pos(
                path=p,
                title=kwargs.get('title', None),
                conf=kwargs.get('conf', None),
                limx=kwargs.get('limx', None),
                limy=kwargs.get('limy', None),
                limz=kwargs.get('limz', None),
                save=kwargs.get('save', None),
                show=False#kwargs.get('show', True)
            )

            self.plot_dir(
                path=p,
                title=kwargs.get('title', None),
                conf=kwargs.get('conf', None),
                limx=kwargs.get('limx', None),
                limy=kwargs.get('limy', None),
                limz=kwargs.get('limz', None),
                save=kwargs.get('save', None),
                show=False#kwargs.get('show', True)
            )

        return fig


class SubAStarBW_(SubAStarBW):

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

    def stepcost(
            self,
            state,
            parent
    ) -> float:

        return self.dist(state, parent)

    def h(
            self,
            state: State
    ) -> float:
        # for backwards direction, the heuristic measures the mean of the distances of the current state's variables
        # and the ones from the initstate. If `state` does not contain all variables of the initstate (which typically
        # applies to the goal state), the distance is infinite.
        if not set(self.initstate.keys()).issubset(set(state.keys())): return np.inf

        return self.dist(state, self.initstate)

    def plot_pos(
            self,
            path: List,
            title: str = None,
            conf: float = None,
            limx: Tuple = None,
            limy: Tuple = None,
            limz: Tuple = None,
            save: str = None,
            show: bool = True
    ) -> Figure:

        # generate datapoints
        data = pd.DataFrame(
                data=[
                    self.gendata(
                        'x_in',
                        'y_in',
                        s,
                        conf=conf
                    ) for s in path if not isinstance(s, self.goal_t)
                ],
                columns=['x_in', 'y_in', 'z', 'lbl']
            )

        if not data.empty:
            return plot_heatmap(
                xvar='x_in',
                yvar='y_in',
                data=data,
                title=title,
                limx=limx,
                limy=limy,
                limz=limz,
                save=save,
                show=show
            )

    def plot_dir(
            self,
            path: List,
            title: str = None,
            conf: float = None,
            limx: Tuple = None,
            limy: Tuple = None,
            limz: Tuple = None,
            save: str = None,
            show: bool = True
    ) -> Figure:

        # generate datapoints
        data = pd.DataFrame(
                data=[
                    self.gendata(
                        'xdir_in',
                        'ydir_in',
                        s,
                        conf=conf
                    ) for s in path if not isinstance(s, self.goal_t)
                ],
                columns=['xdir_in', 'ydir_in', 'z', 'lbl']
            )

        if not data.empty:
            return plot_heatmap(
                xvar='xdir_in',
                yvar='ydir_in',
                data=data,
                title=title,
                limx=limx,
                limy=limy,
                limz=limz,
                save=save,
                show=show
            )

    def plot_path(
            self,
            xvar,
            yvar,
            p: List,
            obstacles: List = None,
            title: str = None,
            save: str = None,
            show: bool = False,
            limx: Tuple = None,
            limy: Tuple = None
    ) -> Figure:

        # generate data points
        d = [
            (
                np.mean([s[xvar].mpe()[0].lower, s[xvar].mpe()[0].upper]),          # x
                np.mean([s[yvar].mpe()[0].lower, s[yvar].mpe()[0].upper]),          # y
                np.mean([s['xdir_in'].mpe()[0].lower, s['xdir_in'].mpe()[0].upper]),     # dx
                np.mean([s['ydir_in'].mpe()[0].lower, s['ydir_in'].mpe()[0].upper]),     # dy
                f'Step {i}',                    # step
                f'<b>Step {i}: {"root" if s.leaf is None or s.tree is None else f"{s.tree}-Leaf#{s.leaf}"}</b><br><b>MPEs:</b><br>{"<br>".join(f"{k}: {fmt(v)}" for k, v in s.items())}',  # lbl
                1                               # size
            )
            for i, s in enumerate(p) if not isinstance(s, Goal)
        ]

        # generate data for scatter circles and quivers
        data = pd.DataFrame(
            data=d,
            columns=[xvar, yvar, 'dx', 'dy', 'step', 'lbl', 'size']
        )

        # generate data for distribution blobs
        data_dists = pd.DataFrame(
                data=[
                    self.gendata(
                        xvar,
                        yvar,
                        s,
                    ) for s in p if not isinstance(s, Goal)
                ],
                columns=[xvar, yvar, 'z', 'lbl']
            )

        # generate data for init dist blob
        data_init = pd.DataFrame(
            data=[
                self.gendata(
                    xvar,
                    yvar,
                    self.initstate
                )
            ],
            columns=[xvar, yvar, 'z', 'lbl']
        )

        # determine corners of goal area
        gxl = self.goal[xvar].lower if isinstance(self.goal[xvar], ContinuousSet) else first(self.goal[xvar])
        gxu = self.goal[xvar].upper if isinstance(self.goal[xvar], ContinuousSet) else first(self.goal[xvar])
        gyl = self.goal[yvar].lower if isinstance(self.goal[yvar], ContinuousSet) else first(self.goal[yvar])
        gyu = self.goal[yvar].upper if isinstance(self.goal[yvar], ContinuousSet) else first(self.goal[yvar])

        # determine position and direction of initstate
        ix = np.mean([self.initstate[xvar].mpe()[0].lower, self.initstate[xvar].mpe()[0].upper])
        iy = np.mean([self.initstate[yvar].mpe()[0].lower, self.initstate[yvar].mpe()[0].upper])
        ixd = np.mean([self.initstate['xdir_in'].mpe()[0].lower, self.initstate['xdir_in'].mpe()[0].upper])
        iyd = np.mean([self.initstate['ydir_in'].mpe()[0].lower, self.initstate['ydir_in'].mpe()[0].upper])

        mainfig = go.Figure()

        # plot obstacles in background
        if obstacles is not None:
            for (o, on) in obstacles:
                mainfig.add_trace(
                    plotly_sq(o, lbl=on, color='rgb(15,21,110)', legend=False))

        fig_initstate = go.Figure()

        # generate heatmap plots for distribution blobs
        fig_path_init = plot_dists_layered(
            xvar,
            yvar,
            data_init,
            limx=limx,
            limy=limy,
            show=False
        )

        # add distribution blobs to main plot
        fig_initstate.add_traces(
            data=fig_path_init.data
        )

        # generate initstate plot
        fig_initstate.add_traces(
            data=plotly_pt(
                pt=(ix, iy),
                dir=(ixd, iyd),
                name=f"Start<br>x_in: {ix}<br>y_in: {iy}",
                color='rgb(0,127,0)'
            ).data
        )

        # generate goal area plot
        fig_initstate.add_trace(
            plotly_sq(
                area=(gxl, gyl, gxu, gyu),
                lbl=f"Goal Area",
                legend=False,
                color='rgb(0,127,0)'
            )
        )

        # add initstate and goal area plots to main plot
        mainfig.add_traces(
            data=fig_initstate.data
        )

        # generate heatmap plots for distribution blobs
        if not data_dists.empty:
            fig_path_dists = plot_dists_layered(
                xvar,
                yvar,
                data_dists,
                limx=limx,
                limy=limy,
                show=False
            )

            # add distribution blobs to main plot
            mainfig.add_traces(
                data=fig_path_dists.data
            )

        # generate scatter/quiver plot for steps
        if not data.empty:
            fig_path = plot_scatter_quiver(
                xvar,
                yvar,
                data,
                show=False
            )

            # add scatter/quiver plot to main plot
            mainfig.add_traces(
                data=fig_path.data
            )

        # set range and size of main plot
        mainfig.update_layout(
            xaxis=dict(
                title=xvar,
                side='top',
                range=[*limx]
            ),
            yaxis=dict(
                title=yvar,
                range=[*limy]
            ),
            height=1000,
            width=1100,
            title=title
        )

        if save:
            fig_to_file(mainfig, save)

        if show:
            mainfig.show(config=defaultconfig(save))

        return mainfig

    def gendata(
            self,
            xvar,
            yvar,
            state,
            conf: float = None
    ):

        # generate datapoints
        x = state[xvar].pdf.boundaries()
        y = state[yvar].pdf.boundaries()

        X, Y = np.meshgrid(x, y)
        Z = np.array(
            [
                state[xvar].pdf(x) * state[yvar].pdf(y)
                for x, y, in zip(X.ravel(), Y.ravel())
            ]).reshape(X.shape)

        # show only values above a certain threshold, consider lower values as high-uncertainty areas
        if conf is not None:
            Z[Z < conf] = 0.

        # remove or replace by eliminating values > median
        Z[Z > np.median(Z)] = np.median(Z)

        lbl = (f'<b>{"ROOT" if state.leaf is None or state.tree is None else f"{state.tree}-Leaf#{state.leaf}"}</b><br>'
               f'<b>MPEs:</b><br>'
               f'{"<br>".join(f"<i>{k}:</i> {fmt(v)}" for k, v in state.items())}<br>'
               f'<b>Expectations:</b><br>'
               f'{"<br>".join(f"<i>{k}:</i> {fmt(v.expectation())}" for k, v in state.items())}<br>')  # lbl

        params = f"<b>Params (MPEs):</b><br>"
        params += "None" if state.tree is None or state.leaf is None else f'{"<br>".join([f"<i>{v.name}:</i> {fmt(self.models.get(state.tree).leaves[state.leaf].distributions[v])}" for v in self.models.get(state.tree).features if v not in state])}'

        return x, y, Z, lbl+params

    def plot(
            self,
            node: Node,
            **kwargs
    ) -> Figure:

        p = self.retrace_path(node)

        obstacles = [
            ((0, 0, 100, 100), "kitchen_boundaries"),
            ((15, 10, 25, 20), "chair1"),
            ((35, 10, 45, 20), "chair2"),
            ((10, 30, 50, 50), "kitchen_island"),
            ((80, 30, 100, 70), "stove"),
            ((10, 80, 50, 100), "kitchen_unit"),
            ((60, 80, 80, 100), "fridge"),
        ]

        fig=self.plot_path(
            xvar='x_in',
            yvar='y_in',
            p=p,
            obstacles=obstacles,
            title=f'SubAStar (bwd)<br>{str(node)}',
            save=os.path.join(locs.logs, f'{os.path.basename(os.path.join(locs.logs, "ASTAR-bwd-path.html"))}'),
            show=True,
            limx=(0, 100),
            limy=(0, 100)
        )

        if kwargs.get('plotdistributions', True):
            # plot distributions
            self.plot_pos(
                path=p,
                title=kwargs.get('title', None),
                conf=kwargs.get('conf', None),
                limx=kwargs.get('limx', None),
                limy=kwargs.get('limy', None),
                limz=kwargs.get('limz', None),
                save=kwargs.get('save', None),
                show=False#kwargs.get('show', True)
            )

            self.plot_dir(
                path=p,
                title=kwargs.get('title', None),
                conf=kwargs.get('conf', None),
                limx=kwargs.get('limx', None),
                limy=kwargs.get('limy', None),
                limz=kwargs.get('limz', None),
                save=kwargs.get('save', None),
                show=False#kwargs.get('show', True)
            )

        return fig
