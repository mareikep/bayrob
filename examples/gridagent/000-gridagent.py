import math
import os
from typing import Tuple

import numpy as np

import dnutils

import pandas as pd
from calo.logs.logs import init_loggers
from calo.utils import locs
from calo.utils.constants import calologger
from jpt import JPT, infer_from_dataframe
from pandas import DataFrame

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)

ACTIONS = [(-1, 0), (0, -1), (1, 0), (0, 1)]
DIRACTIONS = {-90: -1, 0: 0, 90: 1}
ACTIONS_ = ["up", "left", "down", "right"]
OBSTACLES = [
    (1, 1),
    (3, 2),
    (4, 0),
    (4, 2),
    (6, 4),
]


def gendata_move(
        s: int,
        dt: str

) -> Tuple[DataFrame, DataFrame]:
    df_move = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'x_in', 'y_in', 'x_out', 'y_out', 'collided'])
    c = 0

    for y in range(s):
        for x in range(s):
            for i, (y_, x_) in enumerate(ACTIONS):

                # if current position is obstacle skip
                if (y, x) in OBSTACLES:
                    continue

                # if action would move agent into wall or obstacle, do not update position
                collided = not 0 <= x + x_ < s or not 0 <= y + y_ < s or (y + y_, x + x_) in OBSTACLES
                df_move.loc[c] = [
                    x_,  # xdir_in
                    y_,  # ydir_in
                    x,  # x_in
                    y,  # y_in
                    0 if collided else x_,  # x_out (delta)
                    0 if collided else y_,  # y_out (delta)
                    collided  # collided
                ]
                c+=1

    df_move.to_csv(os.path.join(dt, '000-gridagent-move.csv'), index=False)

    return df_move


def gendata_turn(
        s: int,
        dt: str

) -> Tuple[DataFrame, DataFrame]:
    df_turn = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'angle', 'xdir_out', 'ydir_out'])
    for i, (y, x) in enumerate(ACTIONS):
        for j, d in enumerate(DIRACTIONS):
            deg = np.radians(d)
            newdir = x * math.cos(deg) - y * math.sin(deg), x * math.sin(deg) + y * math.cos(deg)
            df_turn.loc[i + j*len(ACTIONS)] = [
                x,  # xdir_in
                y,  # ydir_in
                DIRACTIONS[d],  # angle
                int(newdir[0]) - x,  # xdir_out (delta)
                int(newdir[1]) - y  # ydir_out (delta)
            ]

    df_turn.to_csv(os.path.join(dt, '000-gridagent-turn.csv'), index=False)

    return df_turn


def loaddata(
        d: str,
        dt: str

) -> Tuple[DataFrame, DataFrame]:
    with open(os.path.join(dt, f'000-gridagent-{d}.csv'), 'r') as f:
        df = pd.read_csv(f, delimiter=',', header=0)

    return df


def learn_move(d, dt):
    vars = infer_from_dataframe(d)

    jpt = JPT(
        variables=vars,
        targets=vars[4:],
        min_samples_leaf=1
    )

    jpt.learn(d)
    jpt.postprocess_leaves()

    logger.debug(f'...done! saving to file {os.path.join(dt, f"000-gridagent-move.tree")}')

    jpt.save(os.path.join(dt, f'000-gridagent-move.tree'))
    jpt.plot(
        title=f'Gridagent-move',
        plotvars=list(jpt.variables),
        filename=f'000-gridagent-move',
        directory=dt,
        leaffill='#CCDAFF',
        nodefill='#768ABE',
        alphabet=True,
        view=True
    )


def learn_turn(d, dt):
    vars = infer_from_dataframe(d)

    jpt = JPT(
        variables=vars,
        targets=vars[3:],
        min_samples_leaf=1
    )

    jpt.learn(d, keep_samples=True)
    # jpt.postprocess_leaves()

    logger.debug(f'...done! saving to file {os.path.join(dt, f"000-gridagent-turn.tree")}')

    jpt.save(os.path.join(dt, f'000-gridagent-turn.tree'))
    jpt.plot(
        title=f'Gridagent-turn',
        plotvars=list(jpt.variables),
        filename=f'000-gridagent-turn',
        directory=dt,
        leaffill='#CCDAFF',
        nodefill='#768ABE',
        alphabet=True,
        view=True
    )


if __name__ == "__main__":
    init_loggers(level='debug')

    # use most recently created dataset or create from scratch
    DT = os.path.join(locs.examples, 'gridagent')

    logger.debug(f'running gridagent data generation with data in {DT}')

    # df_move = loaddata("move", DT)
    df_move = gendata_move(7, DT)
    learn_move(df_move, DT)

    # df_turn = loaddata("turn", DT)
    df_turn = gendata_turn(7, DT)
    learn_turn(df_turn, DT)
