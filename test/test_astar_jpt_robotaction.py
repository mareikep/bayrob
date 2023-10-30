import os
import unittest
from pathlib import Path
from typing import Tuple, Any, List

import numpy as np
import pandas as pd
from calo.utils.plotlib import plot_pos, plot_dir, plot_path, plot_tree_dist
from dnutils import ifnone, first

from calo.application.astar_jpt_app import State_, SubAStar_, SubAStarBW_
from calo.core.astar import BiDirAStar
from calo.core.astar_jpt import Goal
from calo.utils import locs
from calo.utils.utils import recent_example
from jpt import JPT
from jpt.base.intervals import ContinuousSet


from matplotlib import pyplot as plt
from jpt.distributions import Numeric, Gaussian


class AStarRobotActionJPTTests(unittest.TestCase):

    @staticmethod
    def plot(
            tree: JPT,
            qvarx: Any = None,
            qvary: Any = None,
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
        x = np.linspace(limx[0], limx[1], 50)
        y = np.linspace(limy[0], limy[1], 50)

        X, Y = np.meshgrid(x, y)
        Z = np.array(
            [
                tree.pdf(
                    tree.bind(
                        {
                            qvarx: x,
                            qvary: y,
                            # tree.varnames['collided']: False
                        }
                    )
                ) for x, y, in zip(X.ravel(), Y.ravel())
                # tree.pdf(
                #     tree.bind(
                #         {
                #             qvarx: x,
                #             qvary: y,
                #             tree.varnames['collided']: False
                #         }
                #     )
                # ) for x, y, in zip(X.ravel(), Y.ravel())
            ]
        ).reshape(X.shape)

        # determine limits
        xmin = ifnone(limx, min(x) - 15, lambda l: l[0])
        xmax = ifnone(limx, max(x) + 15, lambda l: l[1])
        ymin = ifnone(limy, min(y) - 15, lambda l: l[0])
        ymax = ifnone(limy, max(y) + 15, lambda l: l[1])

        # show only values above a certain threshold, consider lower values as high-uncertainty areas
        if conf is not None:
            Z[Z < conf] = 0.

        # remove or replace by eliminating values > median
        # Z[Z > np.median(Z)] = np.median(Z)

        zmin = ifnone(limz, Z.min(), lambda l: l[0])
        zmax = ifnone(limz, Z.max(), lambda l: l[1])

        # init plot
        fig, ax = plt.subplots(num=1, clear=True)
        fig.patch.set_facecolor('#D6E7F8')  # set bg color around the plot area (royal purple)
        ax.set_facecolor('white')  # set bg color of plot area (dark purple)
        cmap = 'BuPu'  # viridis, Blues, PuBu, 0rRd, BuPu

        # generate heatmap
        # c = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=zmin, vmax=zmax)
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none')
        ax.set_title(f'P(x,y)')

        # setting the limits of the plot to the limits of the data
        ax.axis([xmin, xmax, ymin, ymax])
        # ax.axis([-100, 100, -100, 100])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        # fig.colorbar(c, ax=ax)
        fig.suptitle(title)
        fig.canvas.manager.set_window_title(f'Initial distribution: P(x/y)')

        if save:
            plt.savefig(save)

        if show:
            plt.show()

    # assumes trees MOVEFORWARD and TURN generated previously from functions in calo-dev/test/test_robotpos.py,
    # executed in the following order:
    # test_robot_pos_random (xor test_robot_pos) --> generates csv files from consecutive (random) move/turn actions
    # test_data_curation --> curates the previously generaed csv files to one large file each for moveforward and turn
    # actions; this is NOT just a merge of the files but a curation (initial data contains absolute positions or
    # directions, curated data contains deltas!)
    # test_jpt_moveforward --> learns JPT from moveforward data
    # test_jpt_turn --> learns JPT from turn data
    @classmethod
    def setUpClass(cls) -> None:
        recent = recent_example(os.path.join(locs.examples, 'robotaction'))
        # recent = os.path.join(locs.examples, 'robotaction', '2023-08-02_14:23')

        cls.models = dict(
            [
                (
                    treefile.name,
                    JPT.load(str(treefile))
                )
                for p in [recent]
                for treefile in Path(p).rglob('*.tree')
            ]
        )

        # plot initial distributions over x/y positions
        t = cls.models['000-MOVEFORWARD.tree']
        # plot_tree_dist(
        #     tree=t,
        #     qvarx=t.varnames['x_in'],
        #     qvary=t.varnames['y_in'],
        #     title='Initial distribution P(x,y)',
        #     limx=(-100, 100),
        #     limy=(-100, 100),
        #     save=os.path.join(os.path.join(recent, 'plots', '000-init-dist.html')),
        #     show=True
        # )

        tolerance = .01
        # initx, inity, initdirx, initdiry = [-55, 65, 0, -1]
        initx, inity, initdirx, initdiry = [-61, 61, 0, -1]
        # initx, inity, initdirx, initdiry = [-42, 48, 0.5, -.86]

        dx = Gaussian(initx, tolerance).sample(50)
        distx = Numeric()
        distx.fit(dx.reshape(-1, 1), col=0)

        dy = Gaussian(inity, tolerance).sample(50)
        disty = Numeric()
        disty.fit(dy.reshape(-1, 1), col=0)

        ddx = Gaussian(initdirx, tolerance*tolerance).sample(50)
        distdx = Numeric()
        distdx.fit(ddx.reshape(-1, 1), col=0)

        ddy = Gaussian(initdiry, tolerance*tolerance).sample(50)
        distdy = Numeric()
        distdy.fit(ddy.reshape(-1, 1), col=0)

        cls.initstate = State_()
        cls.initstate.update(
            {
                'x_in': distx,
                'y_in': disty,
                'xdir_in': distdx,
                'ydir_in': distdy
            }
        )

        # cls.initstate.plot(show=True)
        goalx, goaly = [-75, 50]
        cls.goal = Goal()
        cls.goal.update(
            {
                'x_in': ContinuousSet(goalx - abs(tolerance * goalx), goalx + abs(tolerance * goalx)),
                'y_in': ContinuousSet(goaly - abs(tolerance * goaly), goaly + abs(tolerance * goaly))
            }
        )

    def test_astar_fw_path(self) -> None:
        self.a_star = SubAStar_(
            AStarRobotActionJPTTests.initstate,
            AStarRobotActionJPTTests.goal,
            models=self.models
        )
        self.path = list(self.a_star.search())

        # generate mapping from path step (=position) to action executed from this position
        self.actions = {}
        for p in self.path:
            self.actions[
                (
                    first(p['y_in'] if isinstance(p['y_in'], set) else p['y_in'].mpe()[1]),
                    first(p['x_in'] if isinstance(p['x_in'], set) else p['x_in'].mpe()[1])
                )
            ] = (
                (
                    first(p['ydir_in'] if isinstance(p['ydir_in'], set) else p['ydir_in'].mpe()[1]),
                    first(p['xdir_in'] if isinstance(p['xdir_in'], set) else p['xdir_in'].mpe()[1])
                ) if 'ydir_in' in p and 'xdir_in' in p else None
            )

    def test_astar_bw_path(self) -> None:
        self.a_star = SubAStarBW_(
            AStarRobotActionJPTTests.initstate,
            AStarRobotActionJPTTests.goal,
            models=self.models
        )
        self.path = list(self.a_star.search())
        self.path.reverse()

        # generate mapping from path step (=position) to action executed from this position
        self.actions = {}
        for p in self.path:
            self.actions[
                (
                    first(p['y_in'] if isinstance(p['y_in'], set) else p['y_in'].mpe()[1]),
                    first(p['x_in'] if isinstance(p['x_in'], set) else p['x_in'].mpe()[1])
                )
            ] = (
                (
                    first(p['ydir_in'] if isinstance(p['ydir_in'], set) else p['ydir_in'].mpe()[1]),
                    first(p['xdir_in'] if isinstance(p['xdir_in'], set) else p['xdir_in'].mpe()[1])
                ) if 'ydir_in' in p and 'xdir_in' in p else None
            )

    def test_astar_bdir_path(self) -> None:
        self.a_star = BiDirAStar(
            SubAStar_,
            SubAStarBW_,
            AStarRobotActionJPTTests.initstate,
            AStarRobotActionJPTTests.goal,
            models=self.models
        )

        self.path = list(self.a_star.search())

        # generate mapping from path step (=position) to action executed from this position
        self.actions = {}
        for p in self.path:
            self.actions[
                (
                    first(p['y_in'] if isinstance(p['y_in'], set) else p['y_in'].mpe()[1]),
                    first(p['x_in'] if isinstance(p['x_in'], set) else p['x_in'].mpe()[1])
                )
            ] = (
                (
                    first(p['ydir_in'] if isinstance(p['ydir_in'], set) else p['ydir_in'].mpe()[1]),
                    first(p['xdir_in'] if isinstance(p['xdir_in'], set) else p['xdir_in'].mpe()[1])
                ) if 'ydir_in' in p and 'xdir_in' in p else None
            )

    @staticmethod
    def generate_steps(
            evidence,
            tree
    ):
        ct = tree.conditional_jpt(
            evidence=tree.bind(
                {k: v for k, v in evidence.items() if k in tree.varnames},
                allow_singular_values=False
            ),
            fail_on_unsatisfiability=False
        )

        return ct, ct.leaves.values()

    def test_astar_cram_path(self) -> None:
        cmds = [
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(8, 10)}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(8, 10)}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(8, 10)}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(8, 10)}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(8, 10)}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(8, 10)}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(8, 10)}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
        ]

        # VARIANT I
        pass

        # VARIANT II: each leaf of the conditional tree represents one possible action
        s = AStarRobotActionJPTTests.initstate
        p = [[s, {}]]
        for cmd in cmds:
            t = AStarRobotActionJPTTests.models[cmd['tree']]

            # generate evidence by using intervals from the 5th percentile to the 95th percentile for each distribution
            evidence = {
                var: ContinuousSet(s[var].ppf(.05), s[var].ppf(.95)) for var in s.keys()
            }

            if cmd["params"] is not None:
                evidence.update(cmd["params"])

            # candidates are all the leaves from the conditional tree
            t_, _ = self.generate_steps(evidence, t)

            # the "best" candidate is the one with the maximum similarity to the evidence state
            # best = None
            # sim = -1
            # for c in candidates:
            #     # type(self[vn]).jaccard_similarity(val, other[vn])
            #     sim_ = min([type(d).jaccard_similarity(s[v.name], d) for v, d in c.distributions.items() if v.name in s])
            #     if sim_ > sim:
            #         best = c
            #         sim = sim_
            best = t_.posterior(variables=t_.targets)

            # create successor state
            s_ = State_()
            s_.update({k: v for k, v in s.items()})
            s_.tree = cmd['tree']
            s_.leaf = None

            # update belief state of potential predecessor
            for vn, d in best.items():
                # vn = v.name
                # generate new distribution by shifting position delta distributions by expectation of position
                # belief state
                if vn.name != vn.name.replace('_in', '_out') and vn.name.replace('_in', '_out') in best:

                    if vn.name in s_:
                        # if the _in variable is already contained in the state, update it by adding the delta
                        # from the leaf distribution
                        nsegments = len(s_[vn.name].cdf.functions)
                        s_[vn.name] = s_[vn.name] + best[vn.name.replace('_in', '_out')]
                    else:
                        nsegments = len(d.pdf.functions)
                        # else save the result of the _in from the leaf distribution shifted by its delta (_out)
                        s_[vn.name] = d + best[vn.name.replace('_in', '_out')]

                    if hasattr(s_[vn.name], 'approximate'):
                        nsegments = min(10, nsegments)
                        s_[vn.name] = s_[vn.name].approximate(
                            n_segments=nsegments,
                            # error_max=.1
                        )

            p.append([s_, cmd['params']])
            s = State_()
            s.update({k: v for k, v in s_.items()})

        plot_path(
            'x_in',
            'y_in',
            p,
            title="Path A to B",
            save=os.path.join(locs.examples, 'robotaction', 'tmp_plots', f'path.svg')
        )

        plot_pos(
            path=p,
            save=os.path.join(locs.examples, 'robotaction', 'tmp_plots', f'posxy.html'),
            show=True
        )

        # plot_dir(
        #     path=p,
        #     save=os.path.join(locs.examples, 'robotaction', 'tmp_plots', f'dirxy.html'),
        #     show=True,
        # )

        # SubAStar_.plot_xyvars(
        #     xvar='x_in',
        #     yvar='y_in',
        #     path=p,
        #     title=f'Position(x,y)',
        #     limx=[-75, -25],
        #     limy=[40, 75],
        #     limz=[0, 0.05],
        #     save=f'test_astar_cram_path_posxy',
        #     show=False
        # )


    def tearDown(self) -> None:
        # draw path steps into grid (use action symbols)
        print()

    @classmethod
    def tearDownClass(cls) -> None:
        print()
