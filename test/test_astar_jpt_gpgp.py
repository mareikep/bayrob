import json

import dnutils
import math
import os
import unittest
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from dnutils import first
from pandas import DataFrame

from bayrob.application.astar_jpt_app import SubAStar_, SubAStarBW_, State_
from bayrob.core.astar import BiDirAStar
from bayrob.core.astar_jpt import Goal
from bayrob.utils import locs
from bayrob.utils.constants import bayroblogger
from jpt import infer_from_dataframe, JPT
from jpt.distributions import Gaussian, Integer, IntegerType

logger = dnutils.getlogger(bayroblogger, level=dnutils.DEBUG)


class GPGP:
    OBSTACLE = 0
    FREE = 1
    STEP = 8
    ACTIONS = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    DIRACTIONS = {-90: -1, 0: 0, 90: 1}

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

    def __init__(
            self,
            dim_row,
            dim_col
    ):
        self.dim_row = dim_row
        self.dim_col = dim_col
        self.grid = np.zeros((dim_row, dim_col))
        self.grid.reshape(-1)

    def tojson(
            self
    ) -> Dict[str, Any]:
        """Convert the GPGP to a json dictionary that can be serialized. """
        return {
            'dim_row': self.dim_row,
            'dim_col': self.dim_col,
            'grid': self.grid.tolist()
        }

    @staticmethod
    def fromjson(
            data: Dict[str, Any]
    ) -> 'GPGP':
        """
        Construct a GPGP instance from a json dict.

        :data:          The JSON dictionary holding the serialized GPGP data as generated by GPGP.tojson()
        """
        gpgp = GPGP(
            data.get('dim_row'),
            data.get('dim_col')
        )
        gpgp.grid = np.array(data.get('grid'))
        return gpgp

    @staticmethod
    def strworld(
            grid,
            legend=True
    ):
        lgnd = f'\n\n{GPGP.REP[GPGP.FREE]} Free cell\n' \
               f'{GPGP.REP[GPGP.OBSTACLE]} Obstacle\n' \
               f'{GPGP.REP[None]} Goal\n' \
               f'{" ".join([GPGP.REP.get(x, GPGP.REP[None]) for x in GPGP.ACTIONS])} Action executed\n'
        if grid is None:
            return lgnd

        world = '   ' + ' '.join([f'{int(i / 10)}' for i in range(grid.shape[1])]) + '\n'
        world += '   ' + ' '.join([f'{int(i % 10)}' for i in range(grid.shape[1])]) + '\n'
        world += '\n'.join(
            [f'{row:02} {" ".join([GPGP.REP.get(grid[row][col], GPGP.REP[None]) for col in range(grid.shape[1])])}' for
             row in range(grid.shape[0])])
        return world + (lgnd if legend else '\n')

    def generate_track(
            self,
            pts
    ):
        for pt in pts:
            numpts = int(Gaussian(50, 3).sample(1))
            row, col = pt
            for _ in range(numpts):
                col_ = max(0, min(self.grid.shape[1] - 1, int(Gaussian(col, 5).sample(1))))
                row_ = max(0, min(self.grid.shape[0] - 1, int(Gaussian(row, 5).sample(1))))
                self.grid[row_][col_] = GPGP.FREE

        return self.grid

    def gendata_move(
            self,
            dt: str

    ) -> Tuple[DataFrame, DataFrame]:
        df_move = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'x_in', 'y_in', 'x_out', 'y_out', 'collided'])
        c = 0

        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                for i, (row_, col_) in enumerate(GPGP.ACTIONS):

                    # if current position is obstacle skip
                    if (row, col) == GPGP.OBSTACLE:
                        continue

                    # if action would move agent into wall or obstacle, do not update position
                    collided = not 0 <= col + col_ < self.grid.shape[1] or not 0 <= row + row_ < self.grid.shape[0] or (
                    row + row_, col + col_) == GPGP.OBSTACLE
                    df_move.loc[c] = [
                        col_,  # xdir_in
                        row_,  # ydir_in
                        col,  # x_in
                        row,  # y_in
                        0 if collided else col_,  # x_out (delta)
                        0 if collided else row_,  # y_out (delta)
                        collided  # collided
                    ]
                    c += 1

        df_move.to_csv(os.path.join(dt, '000-gpgp-move.csv'), index=False)

        return df_move

    def gendata_turn(
            self,
            dt: str

    ) -> Tuple[DataFrame, DataFrame]:
        df_turn = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'angle', 'xdir_out', 'ydir_out'])
        for i, (row, col) in enumerate(GPGP.ACTIONS):
            for j, d in enumerate(GPGP.DIRACTIONS):
                deg = np.radians(d)
                newdir = col * math.cos(deg) - row * math.sin(deg), col * math.sin(deg) + row * math.cos(deg)
                df_turn.loc[i + j * len(GPGP.ACTIONS)] = [
                    col,  # xdir_in
                    row,  # ydir_in
                    GPGP.DIRACTIONS[d],  # angle
                    int(newdir[0]) - col,  # xdir_out (delta)
                    int(newdir[1]) - row  # ydir_out (delta)
                ]

        df_turn.to_csv(os.path.join(dt, '000-gpgp-turn.csv'), index=False)

        return df_turn

    @staticmethod
    def learn_move(d, dt):
        vars = infer_from_dataframe(d)

        jpt = JPT(
            variables=vars,
            targets=vars[4:],
            min_samples_leaf=1
        )

        jpt.learn(d)
        jpt.postprocess_leaves()

        logger.debug(f'...done! saving to file {os.path.join(dt, f"000-gpgp-move.tree")}')

        jpt.save(os.path.join(dt, f'000-gpgp-move.tree'))
        jpt.plot(
            title=f'GPGP-move',
            plotvars=list(jpt.variables),
            filename=f'000-gpgp-move',
            directory=dt,
            leaffill='#CCDAFF',
            nodefill='#768ABE',
            alphabet=True,
            view=True
        )
        return jpt

    @staticmethod
    def learn_turn(d, dt):
        vars = infer_from_dataframe(d)

        jpt = JPT(
            variables=vars,
            targets=vars[3:],
            min_samples_leaf=1
        )

        jpt.learn(d, keep_samples=True)
        # jpt.postprocess_leaves()

        logger.debug(f'...done! saving to file {os.path.join(dt, f"000-gpgp-turn.tree")}')

        jpt.save(os.path.join(dt, f'000-gpgp-turn.tree'))
        jpt.plot(
            title=f'GPGP-turn',
            plotvars=list(jpt.variables),
            filename=f'000-gpgp-turn',
            directory=dt,
            leaffill='#CCDAFF',
            nodefill='#768ABE',
            alphabet=True,
            view=True
        )
        return jpt


class AStarGPGPJPTTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        DT = os.path.join(locs.examples, 'gpgp')
        LOAD_EXISTING = True

        if LOAD_EXISTING:
            with open(os.path.join(locs.examples, 'gpgp', '000-gpgp.json'), 'r') as f:
                t = json.load(f)
                cls.gpgp = GPGP.fromjson(t)

            jpt_move = JPT.load(os.path.join(locs.examples, 'gpgp', '000-gpgp-move.tree'))
            jpt_turn = JPT.load(os.path.join(locs.examples, 'gpgp', '000-gpgp-turn.tree'))
        else:

            cls.gpgp = GPGP(
                20,
                50
            )
            cls.gpgp.generate_track(
                [
                    [5, 0], [7, 5], [10, 5], [11, 6], [12, 10], [14, 10], [16, 12], [15, 14], [12, 15], [12, 10],
                    [12, 15], [10, 20], [11, 21], [10, 25], [8, 27], [7, 30], [7, 35], [5, 40], [8, 42], [8, 45],
                    [10, 50]
                ]
            )

            with open(os.path.join(locs.examples, 'gpgp', '000-gpgp.json'), 'w+') as f:
                json.dump(cls.gpgp.tojson(), f)

            df_move = cls.gpgp.gendata_move(DT)
            df_turn = cls.gpgp.gendata_turn(DT)

            jpt_move = cls.gpgp.learn_move(df_move, DT)
            jpt_turn = cls.gpgp.learn_turn(df_turn, DT)

        cls.models = {
            'move': jpt_move,
            'turn': jpt_turn
        }

        print(
            cls.gpgp.strworld(
                cls.gpgp.grid,
                legend=True
            )
        )

        tolerance = 0
        initx, inity, initdirx, initdiry = [0, 5, 1, 0]

        dx = Gaussian(initx, tolerance).sample(50)
        posxdist = IntegerType('x', 0, cls.gpgp.grid.shape[1])
        distx = posxdist()
        distx.fit(dx.reshape(-1, 1), col=0)

        dy = Gaussian(inity, tolerance).sample(50)
        posydist = IntegerType('y', 0, cls.gpgp.grid.shape[0])
        disty = posydist()
        disty.fit(dy.reshape(-1, 1), col=0)

        ddx = Gaussian(initdirx, tolerance*tolerance).sample(50)
        dirdist = IntegerType('dir', -1, 1)
        distdx = dirdist()
        distdx.fit(ddx.reshape(-1, 1), col=0)

        ddy = Gaussian(initdiry, tolerance*tolerance).sample(50)
        distdy = dirdist()
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
        cls.goal = Goal()
        cls.goal.update(
            {
                'x_in': {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
                'y_in': {49}
            }
        )

        print(
            GPGP.strworld(
                cls.gpgp.grid,
                legend=False
            )
        )

    def test_astar_fw_path(self) -> None:
        self.a_star = SubAStar_(
            AStarGPGPJPTTests.initstate,
            AStarGPGPJPTTests.goal,
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
            AStarGPGPJPTTests.initstate,
            AStarGPGPJPTTests.goal,
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
            AStarGPGPJPTTests.initstate,
            AStarGPGPJPTTests.goal,
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
        res = [
            [
                AStarGPGPJPTTests.gpgp.grid[row][col] if (row, col) not in self.actions else self.actions.get((row, col), None)
                for col in range(len(AStarGPGPJPTTests.gpgp.grid.shape[1]))
            ]
            for row in range(len(AStarGPGPJPTTests.gpgp.grid.shape[0]))
        ]
        print(AStarGPGPJPTTests.strworld(res, legend=False))

    @classmethod
    def tearDownClass(cls) -> None:
        print(cls.gpgp.grid.strworld(None))
