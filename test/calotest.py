import math
import os

import dnutils
import numpy as np
from dnutils import out
from jpt.base.intervals import ContinuousSet
from matplotlib import pyplot as plt

from calo.core.base import CALO
from calo.logs.logs import init_loggers
from calo.utils import locs
from calo.utils.constants import calologger

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

    # goal position
    goal_x = 2 # 2  # 60
    goal_y = 7 # 6  # 20
    # goal_xdir = .6
    # goal_ydir = .6


    factor = .5
    initstate = {
        'x_in': 0,
        'y_in': 0,
        'xdir_in': start_xdir,
        'ydir_in': start_ydir
    }
    q = {
        # 'x_out': ContinuousSet(goal_x - factor*goal_x, goal_x + factor*goal_x),
        'x_out': ContinuousSet(1.4, 1.8),
        # 'y_out': ContinuousSet(goal_y - factor*goal_y, goal_y + factor*goal_y)
        'y_out': ContinuousSet(5.6, 6.2)
        # 'xdir_out': ContinuousSet(goal_xdir - .3, goal_xdir + .3),
        # 'ydir_out': ContinuousSet(goal_ydir - .3, goal_ydir + .3),
    }

    out(q)
    def g(hyp):
        out(hyp.result)
        steps = 0.
        for step in hyp.steps:
            if 'numsteps' in step.leaf.value:
                steps += step.leaf.value["numsteps"].expectation()
            else:
                steps += 1  # constant value for turn step or step.leaf.value["angle"].expectation()
        return steps

    def h(hyp, curq):
        dist = 0.
        orientationdiff = 0.
        for step in hyp.steps:
            if "x_in" in step.leaf.value and "y_in" in step.leaf.value:
                # distance from first step (= "start" of hypothesis) to start state
                dist += abs(step.leaf.value['x_in'].expectation() - 0.) + abs(step.leaf.value['y_in'].expectation() - 0.)
                return dist
            if "xdir_in" in step.leaf.value and "ydir_in" in step.leaf.value:
                # difference in orientation from first step to start state
                rad = math.atan2(step.leaf.value['ydir_in'].expectation() - start_ydir, step.leaf.value['xdir_in'].expectation() - start_xdir)
                deg = abs(math.degrees(rad))
                orientationdiff += deg
                return orientationdiff
        return dist + orientationdiff


    # init calo
    calo = CALO(stepcost=g, heuristic=h)
    # calo.adddatapath(os.path.join(locs.examples, 'robotaction'))
    calo.adddatapath(os.path.join(locs.examples, 'robotaction-mini'))
    calo.query = q
    calo.state = initstate
    calo.strategy = CALO.ASTAR
    calo.infer()

    fig, ax = plt.subplots()

    # plot start position and direction
    ax.scatter(start_x, start_y, marker='*', label='Start', c='k')
    ax.quiver(start_x, start_y, start_xdir, start_ydir, color='k', width=0.001)

    tleaves = []
    mleaves = []
    # plot results
    c = np.random.rand(len(calo.hypotheses))
    X = []
    Y = []
    XDIR = []
    YDIR = []
    lbls = []
    for j, h in enumerate(calo.hypotheses):
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
    ax.quiver([start_x]*len(X), [start_y]*len(Y), XDIR, YDIR, c, width=0.001)

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



#
# import os
# from jpt.trees import JPT
# from calo.utils import locs
# p = os.path.join(locs.examples, 'robotaction-mini', '2022-09-19_08:48-ALL-TURN.tree')
# tree = JPT.load(p)
# l = tree.leaves[26]
# q = {k.name: k.domain.value2label(v) for k,v in l.path.items()}
# tree.expectation(variables=tree.targets, evidence=q)
# dist = l.distributions['xdir_in']
# evset = q['xdir_in']
# dist._p(evset)
#
