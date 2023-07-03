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
DIRACTIONS = [-90, 0, 90]
ACTIONS_ = ["up", "left", "down", "right"]
OBSTACLES = [
    (1, 1),
    (4, 2),
    (5, 0),
    (5, 2),
    (6, 4),
]


def gendata(
        s: int,
        dt: str

) -> Tuple[DataFrame, DataFrame]:
    df_move = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'x_in', 'y_in', 'x_out', 'y_out', 'collided'])
    c = 0

    for y in range(s):
        for x in range(s):
            for i, (y_, x_) in enumerate(ACTIONS):
                # if action would move agent into wall or obstacle, do not update position
                if not 0 <= x + x_ < s or not 0 <= y + y_ < s or (y + y_, x + x_) in OBSTACLES:
                    df_move.loc[c] = [
                        x_,  # xdir_in
                        y_,  # ydir_in
                        x,  # x_in
                        y,  # y_in
                        0,  # x_out (delta)
                        0,  # y_out (delta)
                        True  # collided
                    ]
                else:
                    df_move.loc[c] = [
                        x_,  # xdir_in
                        y_,  # ydir_in
                        x,  # x_in
                        y,  # y_in
                        x_,  # x_out (delta)
                        y_,  # y_out (delta)
                        False  # collided
                    ]
                c+=1

    df_turn = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'angle', 'xdir_out', 'ydir_out'])
    for i, (y, x) in enumerate(ACTIONS):
        for j, d in enumerate(DIRACTIONS):
            deg = np.radians(d)
            newdir = x * math.cos(deg) - y * math.sin(deg), x * math.sin(deg) + y * math.cos(deg)
            df_turn.loc[i + j*len(ACTIONS)] = [
                x,  # xdir_in
                y,  # ydir_in
                d,  # angle
                int(newdir[0]),  # xdir_out (delta)
                int(newdir[1])  # ydir_out (delta)
            ]
    df_move.to_csv(os.path.join(dt, 'gridagent-move.csv'), index=False)
    df_turn.to_csv(os.path.join(dt, 'gridagent-turn.csv'), index=False)
    return df_move, df_turn


def learn_move(d, dt):
    vars = infer_from_dataframe(d)

    jpt = JPT(
        variables=vars,
        targets=vars[4:],
        min_samples_leaf=1
    )

    jpt.learn(d)
    jpt.postprocess_leaves()

    logger.debug(f'...done! saving to file {os.path.join(dt, f"gridagent-MOVE.tree")}')

    jpt.save(os.path.join(dt, f'gridagent-MOVE.tree'))
    jpt.plot(
        title=f'Gridagent-MOVE',
        plotvars=list(jpt.variables),
        filename=f'gridagent-MOVE',
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

    logger.debug(f'...done! saving to file {os.path.join(dt, f"gridagent-TURN.tree")}')

    jpt.save(os.path.join(dt, f'gridagent-TURN.tree'))
    jpt.plot(
        title=f'Gridagent-TURN',
        plotvars=list(jpt.variables),
        filename=f'gridagent-TURN',
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

    df_move, df_turn = gendata(7, DT)
    # learn_move(df_move, DT)
    learn_turn(df_turn, DT)
