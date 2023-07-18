import os
import unittest

from calo.application.astar_jpt_app import SubAStarBW_, State_
from calo.core.astar_jpt import Goal
from calo.utils import locs
from jpt import JPT
from dnutils import first


class GridWorld:
    GRID = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],  # 0 are free path whereas 1's are obstacles
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
    ]

    ACTIONS = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # up, left, right, down
    OBSTACLE = 1
    FREE = 0
    STEP = 8

    # symbol representations for pretty printing
    REP = {
        FREE: '\u2B1C',  # empty box
        OBSTACLE: '\u2B1B',  # filled box
        STEP: '\u22C6',  # star
        (-1, 0): '\u2191',  # arrow up
        (0, -1): '\u2190',  # arrow left
        (1, 0): '\u2193',  # arrow down
        (0, 1): '\u2192',  # arrow right
        None: '\u2666'  # diamond
    }

    def __init__(self):
        pass

    @staticmethod
    def strworld(
            grid,
            legend=True
    ):
        lgnd = f'\n\n{GridWorld.REP[GridWorld.FREE]} Free cell\n' \
               f'{GridWorld.REP[GridWorld.OBSTACLE]} Obstacle\n' \
               f'{GridWorld.REP[None]} Goal\n' \
               f'{" ".join([GridWorld.REP.get(x, GridWorld.REP[None]) for x in GridWorld.ACTIONS])} Action executed\n'
        if grid is None:
            return lgnd

        world = '\n' + '\n'.join([' '.join([GridWorld.REP.get(grid[row][col], GridWorld.REP[None])
                                            for col in range(len(grid[row]))]) for row in range(len(grid))])
        return world + (lgnd if legend else '\n')


class AStarGridworldJPTTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.models = {
            'move': JPT.load(os.path.join(locs.examples, 'gridagent', '000-gridagent-move.tree')),
            'turn': JPT.load(os.path.join(locs.examples, 'gridagent', '000-gridagent-turn.tree')),
        }

        posx = {0}
        posy = {0}
        dirx = {1}
        diry = {0}

        posteriors = cls.models['move'].posterior(
            evidence={
                'x_in': posx,
                'y_in': posy,
                'xdir_in': dirx,
                'ydir_in': diry
            }
        )

        cls.initstate = State_(
            x_in=posteriors['x_in'],
            y_in=posteriors['y_in'],
            xdir_in=posteriors['xdir_in'],
            ydir_in=posteriors['ydir_in']
        )

        cls.goal = Goal(
            x_out={6},
            y_out={6}
        )

        print(GridWorld.strworld(GridWorld.GRID, legend=False))

    def test_astar_path(self) -> None:
        self.a_star = SubAStarBW_(
            AStarGridworldJPTTests.initstate,
            AStarGridworldJPTTests.goal,
            models=self.models
        )
        self.path = self.a_star.search()

        # self.assertTrue(
        #     self.path == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 4), (5, 4), (5, 5),
        #                   (5, 6), (6, 6)] or
        #     self.path == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 3), (3, 4), (4, 4), (5, 4), (5, 5),
        #                   (5, 6), (6, 6)], msg='A* path incorrect')

    def tearDown(self) -> None:
        # generate mapping from path step (=position) to action executed from this position
        path = list(self.path)
        actions = {
            (first(p['y_out']), first(p['x_out'])): (first(p['ydir_out']), first(p['xdir_out']))
            if 'ydir_out' in p and 'xdir_out' in p else None for p in path}

        # draw path steps into grid (use action symbols)
        res = [[GridWorld.GRID[y][x] if (y, x) not in actions else actions.get((y, x), None) for x in
                range(len(GridWorld.GRID))] for y in range(len(GridWorld.GRID[0]))]
        print(GridWorld.strworld(res, legend=False))

    @classmethod
    def tearDownClass(cls) -> None:
        print(GridWorld.strworld(None))
