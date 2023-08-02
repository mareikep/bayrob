import os
import unittest
from pathlib import Path

import numpy as np
from dnutils import first

from calo.application.astar_jpt_app import State_, SubAStar_, SubAStarBW_
from calo.core.astar import BiDirAStar
from calo.core.astar_jpt import Goal
from calo.utils import locs
from calo.utils.utils import recent_example
from jpt import JPT
from jpt.base.intervals import ContinuousSet

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
        recent = os.path.join(locs.examples, 'robotaction', '2023-08-01_14:57')

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

        tolerance = .2

        initx, inity, initdirx, initdiry = [-60, 75, 0, -1]

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

        cls.initstate.plot(show=True)
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

    def tearDown(self) -> None:
        # draw path steps into grid (use action symbols)
        print()

    @classmethod
    def tearDownClass(cls) -> None:
        print()
