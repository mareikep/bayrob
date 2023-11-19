import os
import unittest
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import pandas as pd
from dnutils import ifnone, first
from jpt.base.intervals import ContinuousSet
from matplotlib import pyplot as plt

from calo.application.astar_jpt_app import State_, SubAStar_, SubAStarBW_
from calo.core.astar import BiDirAStar
from calo.core.astar_jpt import Goal
from calo.utils import locs
from calo.utils.plotlib import plot_pos, plot_path, gendata, plot_heatmap, plot_data_subset, plot_tree_dist, \
    plotly_animation, plotly_sq, defaultconfig
from calo.utils.utils import recent_example
from jpt import JPT
from jpt.distributions import Numeric, Gaussian


class AStarRobotActionJPTTests(unittest.TestCase):

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
        # recent = os.path.join(locs.examples, 'robotaction', '2023-11-18_15:24')
        print("loading example", recent)
        cls.recent = recent

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
        #     limx=(0, 100),
        #     limy=(0, 100),
        #     save=os.path.join(os.path.join(recent, 'plots', '000-init-dist.html')),
        #     show=True
        # )

        # initx, inity, initdirx, initdiry = [-42, 48, 0.5, -.86]
        # initx, inity, initdirx, initdiry = [-55, 65, 0, -1]
        # initx, inity, initdirx, initdiry = [-61, 61, 0, 1]
        # initx, inity, initdirx, initdiry = [-61, 61, -1, 0]  # OK
        # initx, inity, initdirx, initdiry = [-61, 61, 1, 0]  # OK
        # initx, inity, initdirx, initdiry = [-61, 61, 1, 0]  # NICHT OK
        # initx, inity, initdirx, initdiry = [-61, 61, 0, 1]  # NICHT OK
        initx, inity, initdirx, initdiry = [30, 78, 0, -1]

        tolerance = .01

        dx = Gaussian(initx, tolerance).sample(50)
        distx = Numeric()
        distx.fit(dx.reshape(-1, 1), col=0)

        dy = Gaussian(inity, tolerance).sample(50)
        disty = Numeric()
        disty.fit(dy.reshape(-1, 1), col=0)

        ddx = Gaussian(initdirx, tolerance).sample(50)
        distdx = Numeric()
        distdx.fit(ddx.reshape(-1, 1), col=0)

        ddy = Gaussian(initdiry, tolerance).sample(50)
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
        return tree.conditional_jpt(
            evidence=tree.bind(
                {k: v for k, v in evidence.items() if k in tree.varnames},
                allow_singular_values=False
            ),
            fail_on_unsatisfiability=False
        )

    def test_astar_single_action_update(self) -> None:
        s0 = AStarRobotActionJPTTests.initstate  # = [-61, 61, 0, -1]
        tm = AStarRobotActionJPTTests.models['000-MOVEFORWARD.tree']
        tt = AStarRobotActionJPTTests.models['000-TURN.tree']

        # plot init position distribution
        d = pd.DataFrame(
            data=[gendata('x_in', 'y_in', s0)],
            columns=['x', 'y', 'z', 'lbl']
        )
        plot_heatmap(
            xvar='x',
            yvar='y',
            data=d,
            title="$\\text{Init Distribution } P(x_{in},y_{in})$",
            limx=(-65, -55),
            limy=(55, 65),
            show=True
        )

        # ACTION I: MOVE ==============================================================
        # get update dist
        evidence = {
            var: ContinuousSet(s0[var].ppf(.05), s0[var].ppf(.95)) for var in s0.keys()
        }
        print('evidence', evidence)
        tm_ = tm.conditional_jpt(
            evidence=tm.bind(
                {k: v for k, v in evidence.items() if k in tm.varnames},
                allow_singular_values=False
            ),
            fail_on_unsatisfiability=False
        )
        dist_u1 = tm_.posterior(variables=tm.targets)

        # plot position update distribution
        d = pd.DataFrame(
            data=[gendata('x_out', 'y_out', dist_u1)],
            columns=['x', 'y', 'z', 'lbl'])
        plot_heatmap(
            xvar='x',
            yvar='y',
            data=d,
            title="$\delta\\text{-Distribution } P(x_{out},y_{out})$",
            limx=(-3, 3),
            limy=(-3, 3),
            show=True
        )

        # update pos
        # create successor state
        s1 = State_()
        s1.update({k: v for k, v in s0.items()})

        nx = min(10, len(s1['x_in'].cdf.functions))
        dist_u1['x_out'] = dist_u1['x_out'].approximate(.3)
        s1['x_in'] = s1['x_in'] + dist_u1['x_out']
        s1['x_in'] = s1['x_in'].approximate(n_segments=nx)

        ny = min(10, len(s1['y_in'].cdf.functions))
        dist_u1['y_out'] = dist_u1['y_out'].approximate(.3)
        s1['y_in'] = s1['y_in'] + dist_u1['y_out']
        s1['y_in'] = s1['y_in'].approximate(n_segments=ny)

        # plot result
        d = pd.DataFrame(
            data=[gendata('x_in', 'y_in', s1)],
            columns=['x', 'y', 'z', 'lbl'])
        plot_heatmap(
            xvar='x',
            yvar='y',
            data=d,
            title="$\\text{Init} + \delta$",
            limx=(-65, -55),
            limy=(55, 65),
            show=True
        )

        # ACTION II: TURN ==============================================================
        # evidence['angle'] = -9

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
            # {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            # {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            # {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(8, 10)}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(8, 10)}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(8, 10)}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(8, 10)}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(8, 10)}},
            # {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(8, 10)}},
            # {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(8, 10)}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            # {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
        ]

        cmds = [
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(8, 10)}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
            {'tree': '000-TURN.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(8, 10)}},
            {'tree': '000-MOVEFORWARD.tree', 'params': {'action': 'move'}},
        ]

        # VARIANT I
        pass

        # VARIANT II: each leaf of the conditional tree represents one possible action
        s = AStarRobotActionJPTTests.initstate
        p = [[s, {}]]
        for cmd in cmds:
            print('cmd', cmd)
            t = AStarRobotActionJPTTests.models[cmd['tree']]

            # generate evidence by using intervals from the 5th percentile to the 95th percentile for each distribution
            evidence = {
                var: ContinuousSet(s[var].ppf(.05), s[var].ppf(.95)) for var in s.keys()
            }

            if cmd["params"] is not None:
                evidence.update(cmd["params"])

            # candidate is the conditional tree
            # t_ = self.generate_steps(evidence, t)
            best = t.posterior(
                variables=t.targets,
                evidence=t.bind({k: v for k, v in evidence.items() if k in t.varnames},
                    allow_singular_values=False
                ),
                fail_on_unsatisfiability=False
            )

            if best is None:
                print('skipping command', cmd, 'unsatisfiable!')
                continue

            # create successor state
            s_ = State_()
            s_.update({k: v for k, v in s.items()})
            s_.tree = cmd['tree']
            s_.leaf = None

            # update belief state of potential predecessor
            for vn, d in best.items():
                outvar = vn.name
                invar = vn.name.replace('_out', '_in')
                if outvar != invar and invar in s_:
                    # if the _in variable is already contained in the state, update it by adding the delta
                    # from the leaf distribution
                    if len(s_[invar].cdf.functions) > 20:
                        s_[invar] = s_[invar].approximate(.2)
                    if len(best[outvar].cdf.functions) > 20:
                        best[outvar] = best[outvar].approximate(.2)
                    s_[invar] = s_[invar] + best[outvar]
                else:
                    s_[invar] = d

                if hasattr(s_[invar], 'approximate'):
                    s_[invar] = s_[invar].approximate(
                        error_max=.2
                    )

            p.append([s_, cmd['params']])
            s = State_()
            s.update({k: v for k, v in s_.items()})

        plot_path(
            'x_in',
            'y_in',
            p,
            title="Path A to B",
            save=os.path.join(locs.logs, f'path.svg')
        )

        plot_pos(
            path=p,
            save=os.path.join(locs.logs, f'posxy.html'),
            show=True
        )

        # plot_dir(
        #     path=p,
        #     save=os.path.join(locs.logs, f'dirxy.html'),
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

    def test_move_till_collision(self) -> None:
        import plotly.graph_objects as go

        print("loading example", AStarRobotActionJPTTests.recent)

        # VARIANT II: each leaf of the conditional tree represents one possible action
        s = AStarRobotActionJPTTests.initstate
        p = [[s, {}]]
        t = AStarRobotActionJPTTests.models['000-MOVEFORWARD.tree']
        for step in range(3):
            print("Step", step)

            # generate evidence by using intervals from the 5th percentile to the 95th percentile for each distribution
            evidence = {
                var: ContinuousSet(s[var].ppf(.05), s[var].ppf(.95)) for var in s.keys() if isinstance(s[var], Numeric)
            }

            # candidate is the conditional tree
            best = t.posterior(
                variables=t.targets,
                evidence=t.bind({k: v for k, v in evidence.items() if k in t.varnames},
                    allow_singular_values=False
                ),
                fail_on_unsatisfiability=False
            )

            if best is None:
                print('skipping at step', step, 'unsatisfiable!')
                continue

            # create successor state
            s_ = State_()
            s_.update({k: v for k, v in s.items()})
            s_.tree = '000-MOVEFORWARD.tree'
            s_.leaf = None

            # update belief state of potential predecessor
            print("Updating new state...")
            for vn, d in best.items():
                outvar = vn.name
                invar = vn.name.replace('_out', '_in')
                if outvar != invar and invar in s_:
                    # if the _in variable is already contained in the state, update it by adding the delta
                    # from the leaf distribution
                    if len(s_[invar].cdf.functions) > 20:
                        s_[invar] = s_[invar].approximate(.2)
                    if len(best[outvar].cdf.functions) > 20:
                        best[outvar] = best[outvar].approximate(.2)

                    print("adding", best[outvar], best[outvar].expectation(), "to", s_[invar], s_[invar].expectation())
                    s_[invar] = s_[invar] + best[outvar]
                else:
                    s_[invar] = d

                if hasattr(s_[invar], 'approximate'):
                    s_[invar] = s_[invar].approximate(
                        error_max=.2
                    )

            p.append([s_, {'action': 'move'}])
            s = State_()
            s.update({k: v for k, v in s_.items()})

        # plot annotated rectangles representing the obstacles and world boundaries
        obstacles = [
            ((15, 10, 25, 20), "chair1"),
            ((35, 10, 45, 20), "chair2"),
            ((10, 30, 50, 50), "kitchen_island"),
            ((80, 30, 100, 70), "stove"),
            ((10, 80, 50, 100), "kitchen_unit"),
            ((60, 80, 80, 100), "fridge"),
        ]

        fig = plot_path(
            'x_in',
            'y_in',
            p,
            title="Path A to B",
            save=os.path.join(locs.logs, f'path.svg'),
            obstacles=obstacles,
            show=True
        )

        fig.write_html(
            os.path.join(locs.logs, f'path.html'),
            config=defaultconfig,
            include_plotlyjs="cdn"
        )

        fig.show(config=defaultconfig)

        # print heatmap representing position distribution update
        plot_pos(
            path=p,
            save=os.path.join(locs.logs, f'posxy.html'),
            show=True
        )

        # plot animation of collision bar chart representing change of collision status
        frames = [s['collided'].plot(view=False).data for (s, _) in p if 'collided' in s]
        plotly_animation(
            data=frames,
            save=os.path.join(locs.logs, f'collision.html'),
            show=True
        )

    def tearDown(self) -> None:
        # draw path steps into grid (use action symbols)
        print()

    @classmethod
    def tearDownClass(cls) -> None:
        print()
