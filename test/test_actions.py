import math
import unittest

from ddt import unpack, data, ddt

from calo.models.action import Move
from calo.models.world import GridAgent, Grid

@ddt
class ThesisActionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        w = Grid(
            x=[0, 30],
            y=[0, 30]
        )
        
        a = GridAgent(
            world=w
        )
        
        cls.world = w
        cls.agent = a

    @data(
        ((0, 1), (-.7, .7)),
        ((-.7, .7), (-1, 0)),
        ((-1, 0), (-.7, -.7)),
        ((-.7, -.7), (0, -1)),
        ((0, -1), (.7, -.7)),
        ((.7, -.7), (1, 0)),
        ((1, 0), (.7, .7)),
        ((.7, .7), (0, 1)),
    )
    @unpack
    def test_turn_left_45deg_free(self, init, exp) -> None:
        self.agent.pos = (20, 20)
        self.agent.dir = init
        deg = -45

        Move.turndeg(self.agent, deg)

        self.assertAlmostEqual(self.agent.dir[0], exp[0], delta=.1, msg="xdir does not match")
        self.assertAlmostEqual(self.agent.dir[1], exp[1], delta=.1, msg="ydir does not match")

    @data(
        ((0, 1), (-1, 0)),
        ((-.7, .7), (-.7, -.7)),
        ((-1, 0), (0, -1)),
        ((-.7, -.7), (.7, -.7)),
        ((0, -1), (1, 0)),
        ((.7, -.7), (.7, .7)),
        ((1, 0), (0, 1)),
        ((.7, .7), (-.7, .7)),
    )
    @unpack
    def test_turn_left_90deg_free(self, init, exp) -> None:
        self.agent.pos = (20, 20)
        self.agent.dir = init
        deg = -90

        Move.turndeg(self.agent, deg)

        self.assertAlmostEqual(self.agent.dir[0], exp[0], delta=.1, msg="xdir does not match")
        self.assertAlmostEqual(self.agent.dir[1], exp[1], delta=.1, msg="ydir does not match")

    @data(
        ((0, 1), (0, -1)),
        ((-.7, .7), (.7, -.7)),
        ((-1, 0), (1, 0)),
        ((-.7, -.7), (.7, .7)),
        ((0, -1), (0, 1)),
        ((.7, -.7), (-.7, .7)),
        ((1, 0), (-1, 0)),
        ((.7, .7), (-.7, -.7)),
    )
    @unpack
    def test_turn_left_180deg_free(self, init, exp) -> None:
        self.agent.pos = (20, 20)
        self.agent.dir = init
        deg = -180

        Move.turndeg(self.agent, deg)

        self.assertAlmostEqual(self.agent.dir[0], exp[0], delta=.1, msg="xdir does not match")
        self.assertAlmostEqual(self.agent.dir[1], exp[1], delta=.1, msg="ydir does not match")

    @data(
        ((0, 1),),
        ((-.7, .7),),
        ((-1, 0),),
        ((-.7, -.7),),
        ((0, -1),),
        ((.7, -.7),),
        ((1, 0),),
        ((.7, .7),),
    )
    @unpack
    def test_turn_left_360deg_free(self, init) -> None:
        self.agent.pos = (20, 20)
        self.agent.dir = init
        deg = -360

        Move.turndeg(self.agent, deg)

        self.assertAlmostEqual(self.agent.dir[0], init[0], delta=.15, msg="xdir does not match")
        self.assertAlmostEqual(self.agent.dir[1], init[1], delta=.15, msg="ydir does not match")

    @data(
        ((0, 1), (.7, .7)),
        ((-.7, .7), (0, 1)),
        ((-1, 0), (-.7, .7)),
        ((-.7, -.7), (-1, 0)),
        ((0, -1), (-.7, -.7)),
        ((.7, -.7), (0, -1)),
        ((1, 0), (.7, -.7)),
        ((.7, .7), (1, 0)),
    )
    @unpack
    def test_turn_right_45deg_free(self, init, exp) -> None:
        self.agent.pos = (20, 20)
        self.agent.dir = init
        deg = 45
        Move.turndeg(self.agent, deg)

        self.assertAlmostEqual(self.agent.dir[0], exp[0], delta=.1, msg="xdir does not match")
        self.assertAlmostEqual(self.agent.dir[1], exp[1], delta=.1, msg="ydir does not match")

    @data(
        ((0, 1), (1, 0)),
        ((-.7, .7), (.7, .7)),
        ((-1, 0), (0, 1)),
        ((-.7, -.7), (-.7, .7)),
        ((0, -1), (-1, 0)),
        ((.7, -.7), (-.7, -.7)),
        ((1, 0), (0, -1)),
        ((.7, .7), (.7, -.7)),
    )
    @unpack
    def test_turn_right_90deg_free(self, init, exp) -> None:
        self.agent.pos = (20, 20)
        self.agent.dir = init
        deg = 90

        Move.turndeg(self.agent, deg)

        self.assertAlmostEqual(self.agent.dir[0], exp[0], delta=.1, msg="xdir does not match")
        self.assertAlmostEqual(self.agent.dir[1], exp[1], delta=.1, msg="ydir does not match")

    @data(
        ((0, 1), (0, -1)),
        ((-.7, .7), (.7, -.7)),
        ((-1, 0), (1, 0)),
        ((-.7, -.7), (.7, .7)),
        ((0, -1), (0, 1)),
        ((.7, -.7), (-.7, .7)),
        ((1, 0), (-1, 0)),
        ((.7, .7), (-.7, -.7)),
    )
    @unpack
    def test_turn_right_180deg_free(self, init, exp) -> None:
        self.agent.pos = (20, 20)
        self.agent.dir = init
        deg = 180

        Move.turndeg(self.agent, deg)

        self.assertAlmostEqual(self.agent.dir[0], exp[0], delta=.1, msg="xdir does not match")
        self.assertAlmostEqual(self.agent.dir[1], exp[1], delta=.1, msg="ydir does not match")

    @data(
        ((0, 1),),
        ((-.7, .7),),
        ((-1, 0),),
        ((-.7, -.7),),
        ((0, -1),),
        ((.7, -.7),),
        ((1, 0),),
        ((.7, .7),),
    )
    @unpack
    def test_turn_right_360deg_free(self, init) -> None:
        self.agent.pos = (20, 20)
        self.agent.dir = init
        deg = 360

        Move.turndeg(self.agent, deg)

        self.assertAlmostEqual(self.agent.dir[0], init[0], delta=.15, msg="xdir does not match")
        self.assertAlmostEqual(self.agent.dir[1], init[1], delta=.15, msg="ydir does not match")
        
        
    # ============================================================

    @data(
        ((0, 1), (18, 21)),
        ((-.7, .7), (19, 20)),
        ((-1, 0), (18, 18)),
        ((-.7, -.7), (20, 18)),
        ((0, -1), (21, 18)),
        ((.7, -.7), (22, 20)),
        ((1, 0), (21, 21)),
        ((.7, .7), (20, 21)),
    )
    @unpack
    def test_turn_left_45deg_move_free(self, init, exp) -> None:
        self.agent.pos = (20, 20)
        self.agent.dir = init
        deg = -45

        Move.turndeg(self.agent, deg)
        Move.moveforward(self.agent)

        self.assertAlmostEqual(self.agent.pos[0], exp[0], delta=1, msg="posx does not match")
        self.assertAlmostEqual(self.agent.pos[1], exp[1], delta=1, msg="posy does not match")

    @data(
        ((0, 1), (18, 20)),
        ((-.7, .7), (18, 18)),
        ((-1, 0), (20, 18)),
        ((-.7, -.7), (21, 18)),
        ((0, -1), (22, 20)),
        ((.7, -.7), (21, 21)),
        ((1, 0), (20, 22)),
        ((.7, .7), (18, 21)),
    )
    @unpack
    def test_turn_left_90deg_move_free(self, init, exp) -> None:
        self.agent.pos = (20, 20)
        self.agent.dir = init
        deg = -90

        Move.turndeg(self.agent, deg)
        Move.moveforward(self.agent)

        self.assertAlmostEqual(self.agent.pos[0], exp[0], delta=1, msg="posx does not match")
        self.assertAlmostEqual(self.agent.pos[1], exp[1], delta=1, msg="posy does not match")

    @data(
        ((0, 1), (20, 18)),
        ((-.7, .7), (21, 18)),
        ((-1, 0), (22, 20)),
        ((-.7, -.7), (21, 21)),
        ((0, -1), (20, 22)),
        ((.7, -.7), (18, 21)),
        ((1, 0), (18, 20)),
        ((.7, .7), (19, 19)),
    )
    @unpack
    def test_turn_left_180deg_move_free(self, init, exp) -> None:
        self.agent.pos = (20, 20)
        self.agent.dir = init
        deg = -180

        Move.turndeg(self.agent, deg)
        Move.moveforward(self.agent)

        self.assertAlmostEqual(self.agent.pos[0], exp[0], delta=1, msg="posx does not match")
        self.assertAlmostEqual(self.agent.pos[1], exp[1], delta=1, msg="posy does not match")

    @data(
        ((0, 1),),
        ((-.7, .7),),
        ((-1, 0),),
        ((-.7, -.7),),
        ((0, -1),),
        ((.7, -.7),),
        ((1, 0),),
        ((.7, .7),),
    )
    @unpack
    def test_turn_left_360deg_move_free(self, init) -> None:
        self.agent.pos = (20, 20)
        self.agent.dir = init
        deg = -360

        Move.turndeg(self.agent, deg)

        self.assertAlmostEqual(self.agent.pos[0], self.agent.pos[0]+init[0], delta=1, msg="posx does not match")
        self.assertAlmostEqual(self.agent.pos[1], self.agent.pos[1]+init[1], delta=1, msg="posy does not match")

    @data(
        ((0, 1), (21, 21)),
        ((-.7, .7), (20, 21)),
        ((-1, 0), (19, 21)),
        ((-.7, -.7), (19, 20)),
        ((0, -1), (19, 19)),
        ((.7, -.7), (20, 19)),
        ((1, 0), (21, 19)),
        ((.7, .7), (21, 20)),
    )
    @unpack
    def test_turn_right_45deg_move_free(self, init, exp) -> None:
        self.agent.pos = (20, 20)
        self.agent.dir = init
        deg = 45
        Move.turndeg(self.agent, deg)

        self.assertAlmostEqual(self.agent.pos[0], exp[0], delta=1, msg="posx does not match")
        self.assertAlmostEqual(self.agent.pos[1], exp[1], delta=1, msg="posy does not match")

    @data(
        ((0, 1), (21, 20)),
        ((-.7, .7), (21, 21)),
        ((-1, 0), (20, 21)),
        ((-.7, -.7), (19, 21)),
        ((0, -1), (19, 20)),
        ((.7, -.7), (19, 19)),
        ((1, 0), (20, 19)),
        ((.7, .7), (21, 19)),
    )
    @unpack
    def test_turn_right_90deg_move_free(self, init, exp) -> None:
        self.agent.pos = (20, 20)
        self.agent.dir = init
        deg = 90

        Move.turndeg(self.agent, deg)

        self.assertAlmostEqual(self.agent.pos[0], exp[0], delta=1, msg="posx does not match")
        self.assertAlmostEqual(self.agent.pos[1], exp[1], delta=1, msg="posy does not match")

    @data(
        ((0, 1), (20, 19)),
        ((-.7, .7), (21, 19)),
        ((-1, 0), (21, 20)),
        ((-.7, -.7), (21, 21)),
        ((0, -1), (20, 21)),
        ((.7, -.7), (19, 21)),
        ((1, 0), (19, 20)),
        ((.7, .7), (19, 19)),
    )
    @unpack
    def test_turn_right_180deg_move_free(self, init, exp) -> None:
        self.agent.pos = (20, 20)
        self.agent.dir = init
        deg = 180

        Move.turndeg(self.agent, deg)

        self.assertAlmostEqual(self.agent.pos[0], exp[0], delta=1, msg="posx does not match")
        self.assertAlmostEqual(self.agent.pos[1], exp[1], delta=1, msg="posy does not match")

    @data(
        ((0, 1),),
        ((-.7, .7),),
        ((-1, 0),),
        ((-.7, -.7),),
        ((0, -1),),
        ((.7, -.7),),
        ((1, 0),),
        ((.7, .7),),
    )
    @unpack
    def test_turn_right_360deg_move_free(self, init) -> None:
        self.agent.pos = (20, 20)
        self.agent.dir = init
        deg = 360

        Move.turndeg(self.agent, deg)

        self.assertAlmostEqual(self.agent.pos[0], self.agent.pos[0]+init[0], delta=1, msg="posx does not match")
        self.assertAlmostEqual(self.agent.pos[1], self.agent.pos[1]+init[1], delta=1, msg="posy does not match")