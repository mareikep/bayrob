import os
import unittest
from pathlib import Path

from jpt.base.intervals import ContinuousSet

from calo.application.astar_jpt_app import SubAStarBW_, SubAStar_, State_
from calo.core.astar_jpt import Goal
from calo.core.astar import BiDirAStar
from calo.utils import locs
from calo.utils.utils import recent_example
from ddt import ddt, data, unpack
from jpt import JPT


@ddt
class CALOAStarAlgorithmTests(unittest.TestCase):
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

    @unittest.skip
    @data(((0, 0, 0, 1), (-10, -10)))
    @unpack
    def test_calofwastar(self, start, goal) -> None:
        # one data set consists of a tuple
        # ((init_pos_x, init_pos_y, startdir_x, startdir_y), (goal_x, goal_y)) of type(s)
        # ((float, float, float, float), (float, float))
        tolerance = .1

        # starting position / initial direction
        posx = ContinuousSet(start[0] - abs(tolerance * start[0]), start[0] + abs(tolerance * start[0]))
        posy = ContinuousSet(start[1] - abs(tolerance * start[1]), start[1] + abs(tolerance * start[1]))
        dirx = ContinuousSet(start[2] - abs(tolerance * start[2]), start[2] + abs(tolerance * start[2]))
        diry = ContinuousSet(start[3] - abs(tolerance * start[3]), start[3] + abs(tolerance * start[3]))

        posteriors = CALOAStarAlgorithmTests.models['000-MOVEFORWARD.tree'].posterior(
            evidence={
                'x_in': posx,
                'y_in': posy,
                'xdir_in': dirx,
                'ydir_in': diry
            }
        )

        initstate = State_(
            posx=posteriors['x_in'],
            posy=posteriors['y_in'],
            dirx=posteriors['xdir_in'],
            diry=posteriors['ydir_in'],
        )

        goalstate = Goal(
            posx=ContinuousSet(goal[0] - abs(tolerance * goal[0]), goal[0] + abs(tolerance * goal[0])),
            posy=ContinuousSet(goal[1] - abs(tolerance * goal[1]), goal[1] + abs(tolerance * goal[1]))
        )

        self.a_star = SubAStar_(
            initstate=initstate,
            goalstate=goalstate,
            models=CALOAStarAlgorithmTests.models,
            state_similarity=.9,
            goal_confidence=1.
        )

        self.path = self.a_star.search()

        self.assertEqual(
            self.path,
            []
        )

    @data(((-75, 75, 0, -1), (-75, 66)))
    @unpack
    def test_calobwastar(self, start, goal) -> None:
        # one data set consists of a tuple
        # ((init_pos_x, init_pos_y, startdir_x, startdir_y), (goal_x, goal_y), models, tolerance) of type(s)
        # ((float, float, float, float), (float, float), List(str), float)
        tolerance = .1

        # starting position / initial direction
        posx = ContinuousSet(start[0] - abs(tolerance * start[0]), start[0] + abs(tolerance * start[0]))
        posy = ContinuousSet(start[1] - abs(tolerance * start[1]), start[1] + abs(tolerance * start[1]))
        dirx = ContinuousSet(start[2] - abs(tolerance * start[2]), start[2] + abs(tolerance * start[2]))
        diry = ContinuousSet(start[3] - abs(tolerance * start[3]), start[3] + abs(tolerance * start[3]))

        posteriors = CALOAStarAlgorithmTests.models['000-MOVEFORWARD.tree'].posterior(
            evidence={
                'x_in': posx,
                'y_in': posy,
                'xdir_in': dirx,
                'ydir_in': diry
            }
        )

        initstate = State_()
        initstate.update(
            {
                'x_in': posteriors['x_in'],
                'y_in': posteriors['y_in'],
                'xdir_in': posteriors['xdir_in'],
                'ydir_in': posteriors['ydir_in']
            }
        )

        goalstate = Goal()
        initstate.update(
            {
                'x_in': ContinuousSet(goal[0] - abs(tolerance * goal[0]), goal[0] + abs(tolerance * goal[0])),
                'y_in': ContinuousSet(goal[1] - abs(tolerance * goal[1]), goal[1] + abs(tolerance * goal[1]))
            }
        )

        self.a_star = SubAStarBW_(
            initstate=initstate,
            goalstate=goalstate,
            models=CALOAStarAlgorithmTests.models,
            state_similarity=.9,
            goal_confidence=1.
        )

        self.path = self.a_star.search()

        self.assertEqual(
            self.path,
            []
        )

    @unittest.skip
    @data(((0, 0, 0, 1), (-10, -10)))
    @unpack
    def test_bdir_caloastar(self, start, goal) -> None:
        # one data set consists of a tuple
        # ((init_pos_x, init_pos_y, startdir_x, startdir_y), (goal_x, goal_y), models, tolerance) of type(s)
        # ((float, float, float, float), (float, float), List(str), float)
        tolerance = .1

        # starting position / initial direction
        posx = ContinuousSet(start[0] - abs(tolerance * start[0]), start[0] + abs(tolerance * start[0]))
        posy = ContinuousSet(start[1] - abs(tolerance * start[1]), start[1] + abs(tolerance * start[1]))
        dirx = ContinuousSet(start[2] - abs(tolerance * start[2]), start[2] + abs(tolerance * start[2]))
        diry = ContinuousSet(start[3] - abs(tolerance * start[3]), start[3] + abs(tolerance * start[3]))

        posteriors = CALOAStarAlgorithmTests.models['000-MOVEFORWARD.tree'].posterior(
            evidence={
                'x_in': posx,
                'y_in': posy,
                'xdir_in': dirx,
                'ydir_in': diry
            }
        )

        initstate = State_(
            posx=posteriors['x_in'],
            posy=posteriors['y_in'],
            dirx=posteriors['xdir_in'],
            diry=posteriors['ydir_in'],
        )

        goalstate = Goal(
            posx=ContinuousSet(goal[0] - abs(tolerance * goal[0]), goal[0] + abs(tolerance * goal[0])),
            posy=ContinuousSet(goal[1] - abs(tolerance * goal[1]), goal[1] + abs(tolerance * goal[1]))
        )

        self.a_star = BiDirAStar(
            f_astar=SubAStar_,
            b_astar=SubAStarBW_,
            initstate=initstate,
            goalstate=goalstate,
            models=CALOAStarAlgorithmTests.models,
            state_similarity=.9,
            goal_confidence=1.
        )

        self.path = self.a_star.search()

        self.assertEqual(
            self.path,
            []
        )

    def tearDown(self) -> None:
        print('tearDown', self.path)
        self.a_star.plot(self.path[-1])
