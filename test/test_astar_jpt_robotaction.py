import os
import unittest
from pathlib import Path

import pandas as pd
from dnutils import first
from jpt.base.intervals import ContinuousSet

from calo.application.astar_jpt_app import State_, SubAStar_, SubAStarBW_
from calo.core.astar import BiDirAStar
from calo.core.astar_jpt import Goal
from calo.utils import locs
from calo.utils.plotlib import gendata, plot_heatmap
from calo.utils.utils import recent_example
from jpt import JPT
from jpt.distributions import Numeric, Gaussian


class AStarRobotActionJPTTests(unittest.TestCase):

    # assumes trees robotaction_move and robotaction_move generated previously from functions in calo-dev/test/test_robotpos.py,
    # executed in the following order:
    # test_robot_pos_random (xor test_robot_pos) --> generates csv files from consecutive (random) move/turn actions
    # test_data_curation --> curates the previously generaed csv files to one large file each for robotaction_move and turn
    # actions; this is NOT just a merge of the files but a curation (initial data contains absolute positions or
    # directions, curated data contains deltas!)
    # test_jpt_robotaction_move --> learns JPT from robotaction_move data
    # test_jpt_turn --> learns JPT from turn data
    @classmethod
    def setUpClass(cls) -> None:
        cls.recent_move = recent_example(os.path.join(locs.examples, 'robotaction_move'))
        cls.recent_turn = recent_example(os.path.join(locs.examples, 'robotaction_turn'))
        # recent = os.path.join(locs.examples, 'robotaction_move', '2023-11-18_15:24')
        print("loading examples from", cls.recent_move, cls.recent_turn)

        cls.models = dict(
            [
                (
                    treefile.name,
                    JPT.load(str(treefile))
                )
                for p in [cls.recent_move, cls.recent_turn]
                for treefile in Path(p).rglob('*.tree')
            ]
        )

        # plot initial distributions over x/y positions
        t = cls.models['000-robotaction_move.tree']
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
        initx, inity, initdirx, initdiry = [20, 70, -1, 0]

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
        tm = AStarRobotActionJPTTests.models['000-robotaction_move.tree']
        tt = AStarRobotActionJPTTests.models['000-robotaction_move.tree']

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
            limx=(0, 100),
            limy=(0, 100),
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
            limx=(0, 100),
            limy=(0, 100),
            show=True
        )

        # ACTION II: robotaction_move ==============================================================
        # evidence['angle'] = -9

    def tearDown(self) -> None:
        # draw path steps into grid (use action symbols)
        print()

    @classmethod
    def tearDownClass(cls) -> None:
        print()
