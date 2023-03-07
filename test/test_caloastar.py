import os
from pathlib import Path

import dnutils
import numpy as np
from matplotlib import pyplot as plt

from calo.core.caloastar import CALOAStar
from calo.core.hypothesis_robot import Hypothesis_Robot
from calo.logs.logs import init_loggers
from calo.utils import locs
from calo.utils.constants import calologger
from jpt import JPT
from jpt.base.intervals import ContinuousSet

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


def runcalo():
    # assumes trees MOVEFORWARD and TURN generated previously from functions in calo-dev/test/test_robotpos.py,
    # executed in the following order:
    # test_robot_pos_random (xor test_robot_pos) --> generates csv files from consecutive (random) move/turn actions
    # test_data_curation --> curates the previously generaed csv files to one large file each for moveforward and turn
    # actions; this is NOT just a merge of the files but a curation (initial data contains absolute positions or
    # directions, curated data contains deltas!)
    # test_jpt_moveforward --> learns JPT from moveforward data
    # test_jpt_turn --> learns JPT from turn data

    # starting position / initial direction
    start_x = 0
    start_y = 0
    start_xdir = 0
    start_ydir = 1

    # goal position: ex. (-16, -9)
    # goal_x = -16
    # goal_y = -9
    goal_x = -10
    goal_y = -10

    start = {
        'x_in': start_x,
        'y_in': start_y,
        'xdir_in': start_xdir,
        'ydir_in': start_ydir
    }

    goal = {
        'x_out': goal_x,
        'y_out': goal_y,
    }

    models = dict([(treefile.name, JPT.load(str(treefile))) for p in [os.path.join(locs.examples, 'robotaction', '2022-09-14_08:44')] for treefile in Path(p).rglob('*.tree')])
    tolerance = .3

    goal_ = {
        'x_out': ContinuousSet(goal['x_out'] - abs(tolerance * goal['x_out']),
                               goal['x_out'] + abs(tolerance * goal['x_out'])),
        'y_out': ContinuousSet(goal['y_out'] - abs(tolerance * goal['y_out']),
                               goal['y_out'] + abs(tolerance * goal['y_out']))
    }

    # init calo
    castar = CALOAStar(Hypothesis_Robot, start, goal_, models=models)
    pathb = castar.search()
    print(pathb)

    # bidir_astar = BiDirAStar(CALOAStar, init, goal, g=g, h=h)
    # pathb = bidir_astar.search()
    # print(pathb)

    # calo = CALO(CALOAStar, initstate, q, datapaths=[os.path.join(locs.examples, 'robotaction', '2022-09-14_08:44')], stepcost=g, heuristic=h)
    # calo.infer()

    fig, ax = plt.subplots()

    # plot start position and direction
    ax.scatter(start_x, start_y, marker='*', label='Start', c='k')
    ax.quiver(start_x, start_y, start_xdir, start_ydir, color='k', width=0.001)

    tleaves = []
    mleaves = []
    # plot results
    c = np.random.rand(len(pathb))
    X = []
    Y = []
    XDIR = []
    YDIR = []
    lbls = []
    for j, h in enumerate(pathb):
        x = start_x + h.result['x_out'].result
        y = start_y + h.result['y_out'].result
        dirx = start_x + h.result['xdir_out'].result
        diry = start_y + h.result['ydir_out'].result

        # generate plot legend labels
        steps = f'{len(h.steps)} Steps: '
        for i, s in enumerate(h.steps):
            if "MOVEFORWARD" in s.treename:
                steps += f'{i}: MOVE {s.leaf.value["numsteps"].expectation():.2f} STEPS (Leaf {s.leaf.idx}); '
                mleaves.append(s.leaf.idx)
            else:
                steps += f'{i}: TURN {s.leaf.value["angle"].expectation():.2f} DEGREES (Leaf {s.leaf.idx}); '
                tleaves.append(s.leaf.idx)

        X.append(x)
        Y.append(y)
        XDIR.append(start_x - dirx)
        YDIR.append(start_y - diry)
        lbls.append(steps)

    ax.scatter(X, Y, marker='*', label=lbls, c=c)
    ax.quiver([start_x]*len(X), [start_y]*len(Y), XDIR, YDIR, c, width=0.001)  # FIXME: direction vector needs to be relative!

    for i, txt in enumerate(lbls):
        ax.annotate(txt, (X[i], Y[i]))

    print('---------------------------------------------------------------')
    print('Turn leaves', set(sorted(tleaves)))
    print('Move leaves', set(sorted(mleaves)))
    print('STEPS', ',\n'.join(lbls))
    # plot goal position and direction
    ax.scatter(goal_x, goal_y, marker='*', label='Goal', c='green')
    ax.quiver(goal_x, goal_y, goal_x-start_x, goal_y-start_y, color='green', width=0.001)

    plt.grid()
    plt.xlim(-2, 100)
    plt.ylim(-2, 100)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    init_loggers(level='debug')
    runcalo()
