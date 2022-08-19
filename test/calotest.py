import dnutils
import numpy as np
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
    xdir = 0
    ydir = 1

    # goal position
    goal_x = 5
    goal_y = 5

    # query is difference from goal and start
    deltax = goal_x-start_x
    deltay = goal_y-start_y

    deltaxdir = goal_x - xdir
    deltaydir = goal_y - ydir

    normfactor = np.linalg.norm([deltaxdir, deltaydir])
    deltaxdirnorm = deltaxdir / normfactor
    deltaydirnorm = deltaydir / normfactor

    # init calo
    calo = CALO()
    calo.adddatapath(locs.logs)
    calo.reloadmodels()

    q = {
        'deltax': ContinuousSet(deltax - 2.5, deltax + 2.5),
        'deltay': ContinuousSet(deltay - 2.5, deltay + 2.5),
        'deltaxdir': ContinuousSet(deltaxdirnorm - 2.5, deltaxdirnorm + 2.5),
        'deltaydir': ContinuousSet(deltaydirnorm - 2.5, deltaydirnorm + 2.5),
    }

    calo.query = q
    calo.strategy = CALO.BFS
    calo.infer()

    # plot start position and direction
    plt.scatter(start_x, start_y, marker='*', label='Start', c='k')
    plt.quiver(start_x, start_y, xdir, ydir, label='init dir', color='k')

    # plot results
    for h in calo.hypotheses:
        x = start_x + h.result['deltax'].result
        y = start_y + h.result['deltay'].result

        # generate plot legend labels
        steps = f'{len(h.steps)} Steps: '
        for i, s in enumerate(h.steps):
            if "MOVEFORWARD" in s.treename:
                steps += f'{i}: MOVE {s.leaf.value["numsteps"].expectation():.2f} STEPS (Leaf {s.leaf.idx}); '
            else:
                steps += f'{i}: TURN {s.leaf.value["angle"].expectation():.2f} DEGREES (Leaf {s.leaf.idx}); '
        plt.scatter(x, y, marker='*', label=steps)

    # plot goal position and direction
    plt.scatter(goal_x, goal_y, marker='*', label='Goal', c='green')
    plt.quiver(goal_x, goal_y, deltaxdir, deltaydir, label='goal dir', color='green')

    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    init_loggers(level='debug')
    runcalo()






