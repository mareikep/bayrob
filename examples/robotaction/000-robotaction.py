import datetime
import os
from random import randint

import dnutils
import numpy as np
import pandas as pd
from calo.utils.utils import recent_example
from jpt.distributions import Gaussian
from matplotlib import pyplot as plt, patches

from calo.logs.logs import init_loggers
from calo.models.action import Move
from calo.models.world import GridAgent, Grid
from calo.utils import locs
from calo.utils.constants import FILESTRFMT, calologger
from jpt import infer_from_dataframe, JPT

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


def robot_pos_random(dt):
    logger.debug('Generating random robot data...')

    # init agent and world
    w = Grid()
    w.obstacle(25, 25, 50, 50)
    w.obstacle(-10, 10, 0, 40)
    w.obstacle(50, -30, 20, 10)
    w.obstacle(-75, -10, -50, -40)
    w.obstacle(-25, -50, -15, -75)

    gaussian_deg = Gaussian(0, 360)

    fig, ax = plt.subplots(num=1, clear=True)

    # write sample data for MOVEFORWARD and TURN action of robot (absolute positions)
    for j in range(RUNS):
        poses = []  # for plotting
        turns = []

        # init agent at random position
        a = GridAgent(world=w)
        a.init_random()

        # for each run and action select random
        for _ in range(NUMACTIONS):
            deg = gaussian_deg.sample(1)
            turns.append(a.dir + (deg,))
            Move.turndeg(a, deg)

            steps = randint(1, 10)
            for _ in range(steps):
                poses.append(a.pos + a.dir + (1, a.collided))
                Move.moveforward(a, 1)

        poses.append(a.pos + a.dir + (0, a.collided))
        turns.append(a.dir + (0,))

        df_moveforward = pd.DataFrame(poses, columns=['x', 'y', 'xdir', 'ydir', 'numsteps', 'collided'])
        df_moveforward.to_csv(os.path.join(locs.examples, 'robotaction', dt, 'data', f'{j}-MOVEFORWARD.csv'), index=False)

        df_turn = pd.DataFrame(turns, columns=['xdir', 'ydir', 'angle'])
        df_turn.to_csv(os.path.join(locs.examples, 'robotaction', dt, 'data', f'{j}-TURN.csv'), index=False)

        # plot trajectories and highlight start and end points
        ax.plot(df_moveforward['x'], df_moveforward['y'], c='cornflowerblue')
        ax.scatter(df_moveforward['x'].iloc[0], df_moveforward['y'].iloc[0], marker='+', c='green', zorder=1000)
        ax.scatter(df_moveforward['x'].iloc[-1], df_moveforward['y'].iloc[-1], marker='+', c='red', zorder=1000)

        # TODO: remove to save storage space and prevent overload of produced images
        plt.savefig(os.path.join(locs.examples, 'robotaction', dt, 'plots', f'{j}-MOVE.png'), format="png")

    # plot annotated rectangles representing the obstacles
    for i, o in enumerate(w.obstacles):
        ax.add_patch(patches.Rectangle(
            (
                o[0],
                o[1]
            ),
            width=o[2] - o[0],
            height=o[3] - o[1],
            linewidth=1,
            color='green',
            alpha=.2)
        )
        ax.annotate(
            f'O{i+1}',
            (
                o[0] + (o[2] - o[0]) / 2,
                o[1] + (o[3] - o[1]) / 2
            )
        )

    logger.debug('...done! Saving plot...')

    # figure settings
    fig.suptitle(f'{RUNS} runs; {NUMACTIONS} actions per run')
    fig.canvas.manager.set_window_title(f'000-ALL-MOVE.png')
    plt.legend()
    plt.grid()

    # save and show
    plt.savefig(os.path.join(locs.examples, 'robotaction', dt, 'plots', f'000-ALL-MOVE.png'), format="png")

    if SHOWPLOTS:
        plt.show()


def robot_pos_semi_random(dt):
    # randomly select position in grid and then move 1 step in #NUMACTIONS different directions
    logger.debug('Generating star-shaped robot data...')

    # init agent and world
    w = Grid()
    w.obstacle(25, 25, 50, 50)
    w.obstacle(-10, 10, 0, 40)
    w.obstacle(50, -30, 20, 10)
    w.obstacle(-75, -10, -50, -40)
    w.obstacle(-25, -50, -15, -75)

    fig, ax = plt.subplots(num=1, clear=True)

    # init agent at left lower corner facing right
    a = GridAgent(
        world=w
    )
    deg_i = 360/16  # intended degree of turn
    g_deg_i = Gaussian(deg_i, 5)

    # write sample data for MOVEFORWARD and TURN action of robot (absolute positions)
    cnt = 0

    # data_moveforward = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'x_in', 'y_in', 'x_out', 'y_out', 'collided'])
    # data_turn = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'angle', 'xdir_out', 'ydir_out'])
    dm_ = []
    dt_ = []

    for y in range(-100, 100, 1):
        for x in range(-100, 100, 1):
            if x == -100:
                print(f'x/y: {x}/{y}')

            npos = (Gaussian(x, .3).sample(1), Gaussian(y, .3).sample(1))

            # do not position agent on obstacles
            if w.collides(npos):
                continue

            a.pos = npos
            a.dir = (1., 0.)
            initpos = a.pos

            # for each position, turn 16 times in positive and negative direction
            for _ in range(16):

                degi = g_deg_i.sample(1)

                adir = a.dir  # save dir before update
                Move.turndeg(a, degi)
                # data_turn.loc[cnt] = [
                #     *adir,
                #     degi,
                #     *np.array(a.dir) - np.array(adir)  # deltas!
                # ]
                dt_.append(
                    [
                        *adir,
                        degi,
                        *np.array(a.dir) - np.array(adir)  # deltas!
                    ]
                )

                Move.moveforward(a, 1)
                # data_moveforward.loc[cnt] = [
                #     *a.dir,
                #     *initpos,
                #     *np.array(a.pos) - np.array(initpos),  # deltas!
                #     a.collided
                # ]
                dm_.append(
                    [
                        *a.dir,
                        *initpos,
                        *np.array(a.pos) - np.array(initpos),  # deltas!
                        a.collided
                    ]
                )

                # reset position
                a.pos = initpos
                cnt += 1

            for _ in range(16):

                degi = -g_deg_i.sample(1)

                adir = a.dir  # save dir before update
                Move.turndeg(a, degi)
                # data_turn.loc[cnt] = [
                #     *adir,
                #     degi,
                #     *a.dir
                # ]
                dt_.append(
                    [
                        *adir,
                        degi,
                        *np.array(a.dir) - np.array(adir)  # deltas!
                    ]
                )

                Move.moveforward(a, 1)
                # data_moveforward.loc[cnt] = [
                #     *a.dir,
                #     *initpos,
                #     *a.pos,
                #     a.collided
                # ]
                dm_.append(
                    [
                        *a.dir,
                        *initpos,
                        *np.array(a.pos) - np.array(initpos),  # deltas!
                        a.collided
                    ]
                )

                # reset position
                a.pos = initpos
                cnt += 1

    data_moveforward = pd.DataFrame(data=dm_, columns=['xdir_in', 'ydir_in', 'x_in', 'y_in', 'x_out', 'y_out', 'collided'])
    data_turn = pd.DataFrame(data=dt_, columns=['xdir_in', 'ydir_in', 'angle', 'xdir_out', 'ydir_out'])

    # save data
    data_moveforward = data_moveforward.astype({
        'xdir_in': np.float64,
        'ydir_in': np.float64,
        'x_in': np.float64,
        'y_in': np.float64,
        'x_out': np.float64,
        'y_out': np.float64,
        'collided': bool
    })
    data_moveforward.to_csv(os.path.join(locs.examples, 'robotaction', dt, 'data', f'000-ALL-MOVEFORWARD.csv'), index=False)

    data_turn = data_turn.astype({
        'xdir_in': np.float64,
        'ydir_in': np.float64,
        'xdir_out': np.float64,
        'ydir_out': np.float64,
        'angle': np.float64
    })
    data_turn.to_csv(os.path.join(locs.examples, 'robotaction', dt, 'data', f'000-ALL-TURN.csv'), index=False)

    # plot trajectories and highlight start and end points
    c = np.arange(data_moveforward.shape[0])
    ax.scatter(
        data_moveforward['x_in'],
        data_moveforward['y_in'],
        marker='+',
        c=c,
        alpha=.005
    )

    # TODO: remove to save storage space and prevent overload of produced images
    plt.savefig(os.path.join(locs.examples, 'robotaction', dt, 'plots', f'{cnt}-MOVE.png'), format="png")

    # plot annotated rectangles representing the obstacles
    for i, o in enumerate(w.obstacles):
        ax.add_patch(patches.Rectangle(
            (
                o[0],
                o[1]
            ),
            width=o[2] - o[0],
            height=o[3] - o[1],
            linewidth=1,
            color='green',
            alpha=.2)
        )
        ax.annotate(
            f'O{i+1}',
            (
                o[0] + (o[2] - o[0]) / 2,
                o[1] + (o[3] - o[1]) / 2
            )
        )

    logger.debug('...done! Saving plot...')

    # figure settings
    fig.suptitle(f'{RUNS} runs; {NUMACTIONS} actions per run')
    fig.canvas.manager.set_window_title(f'000-ALL-MOVE.png')
    plt.legend()
    plt.grid()

    # save and show
    plt.savefig(os.path.join(locs.examples, 'robotaction', dt, 'plots', f'000-ALL-MOVE.png'), format="png")

    if SHOWPLOTS:
        plt.show()


def data_curation(dt, usedeltas=False):
    logger.debug('curating robot MOVEFORWARD data...')

    # read position data files generated by test_robot_pos and generate large file containing deltas
    # (position-independent)

    if COLLIDED:
        data_moveforward = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'x_in', 'y_in', 'x_out', 'y_out', 'collided'])
    else:
        data_moveforward = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'x_in', 'y_in', 'x_out', 'y_out'])

    for i in range(RUNS):
        with open(os.path.join(locs.examples, 'robotaction', dt, 'data', f'{i}-MOVEFORWARD.csv'), 'r') as f:
            logger.debug(f"Reading {os.path.join(locs.examples, 'robotaction', dt, 'data', f'{i}-MOVEFORWARD.csv')}...")
            d = pd.read_csv(f, delimiter=',', header=0)
            for idx, row in d.iterrows():
                if idx == d.index.max(): break

                if COLLIDED:
                    data_moveforward.loc[idx + i * RUNS] = [
                        row['xdir'],
                        row['ydir'],
                        row['x'],
                        row['y'],
                        d.iloc[idx + 1]['x']-row['x'] if usedeltas else d.iloc[idx + 1]['x'],
                        d.iloc[idx + 1]['y']-row['y'] if usedeltas else d.iloc[idx + 1]['y'],
                        row['collided']
                    ]
                else:
                    data_moveforward.loc[idx + i * RUNS] = [
                        row['xdir'],
                        row['ydir'],
                        row['x'],
                        row['y'],
                        d.iloc[idx + 1]['x']-row['x'] if usedeltas else d.iloc[idx + 1]['x'],
                        d.iloc[idx + 1]['y']-row['y'] if usedeltas else d.iloc[idx + 1]['y']
                    ]

    data_moveforward.to_csv(os.path.join(locs.examples, 'robotaction', dt, 'data', f'000-ALL-MOVEFORWARD.csv'), index=False)

    logger.debug('...done! curating robot TURN data...')
    data_turn = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'angle', 'xdir_out', 'ydir_out'])
    for i in range(RUNS):
        with open(os.path.join(locs.examples, 'robotaction', dt, 'data', f'{i}-TURN.csv'), 'r') as f:
            logger.debug(f"Reading {os.path.join(locs.examples, 'robotaction', dt, 'data', f'{i}-TURN.csv')}...")
            d = pd.read_csv(f, delimiter=',', header=0)
            for idx, row in d.iterrows():
                if idx == d.index.max(): continue
                # store data of two consecuting steps
                data_turn.loc[idx + i * RUNS] = [
                    row['xdir'],
                    row['ydir'],
                    row['angle'],
                    d.iloc[idx + 1]['xdir']-row['xdir'] if usedeltas else d.iloc[idx + 1]['xdir'],
                    d.iloc[idx + 1]['ydir']-row['ydir'] if usedeltas else d.iloc[idx + 1]['ydir']
                ]

    data_turn.to_csv(os.path.join(locs.examples, 'robotaction', dt, 'data', f'000-ALL-TURN.csv'), index=False)
    logger.debug('...done!')


def data_curation_semi(dt):
    logger.debug('curating semi robot MOVEFORWARD data...')

    # read position data files generated by test_robot_pos and generate large file containing deltas
    # (position-independent)

    if COLLIDED:
        data_moveforward = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'x_in', 'y_in', 'x_out', 'y_out', 'collided'])
    else:
        data_moveforward = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'x_in', 'y_in', 'x_out', 'y_out'])

    for i in range(1):
        with open(os.path.join(locs.examples, 'robotaction', dt, 'data', f'{i}-MOVEFORWARD.csv'), 'r') as f:
            d = pd.read_csv(f, delimiter=',', header=0)
            for idx, row in d.iterrows():
                if idx == d.index.max(): break

                if COLLIDED:
                    data_moveforward.loc[idx + i * RUNS] = [
                        row['xdir'],
                        row['ydir'],
                        d.iloc[0]['x'],  # first x-value in file is always starting point
                        d.iloc[0]['y'],  # first y-value in file is always starting point
                        row['x'],
                        row['y'],
                        row['collided']
                    ]
                else:
                    data_moveforward.loc[idx + i * RUNS] = [
                        row['xdir'],
                        row['ydir'],
                        d.iloc[0]['x'],  # first x-value in file is always starting point
                        d.iloc[0]['y'],  # first y-value in file is always starting point
                        row['x'],
                        row['y']
                    ]

    data_moveforward.to_csv(os.path.join(locs.examples, 'robotaction', dt, 'data', f'000-ALL-MOVEFORWARD.csv'), index=False)

    logger.debug('...done! curating semi robot TURN data...')
    data_turn = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'angle', 'xdir_out', 'ydir_out'])
    for i in range(RUNS):
        with open(os.path.join(locs.examples, 'robotaction', dt, 'data', f'{i}-TURN.csv'), 'r') as f:
            d = pd.read_csv(f, delimiter=',', header=0)
            for idx, row in d.iterrows():
                if idx == d.index.max(): continue
                # store data of two consecuting steps
                data_turn.loc[idx + i * RUNS] = [
                    row['xdir'],
                    row['ydir'],
                    row['angle'],
                    d.iloc[idx + 1]['xdir'],
                    d.iloc[idx + 1]['ydir']
                ]

    data_turn.to_csv(os.path.join(locs.examples, 'robotaction', dt, 'data', f'000-ALL-TURN.csv'), index=False)
    logger.debug('...done!')


def learn_jpt_moveforward(dt):
    logger.debug('learning MOVEFORWARD JPT...')

    # learn discriminative JPT from data generated by test_data_curation for MOVEFORWARD
    data_moveforward = pd.read_csv(
        os.path.join(locs.examples, 'robotaction', dt, 'data', f'000-ALL-MOVEFORWARD.csv'),
        delimiter=',',
        header=0
    )
    movevars = infer_from_dataframe(data_moveforward, scale_numeric_types=False)

    jpt_mf = JPT(
        variables=movevars,
        targets=movevars[4:],
        min_impurity_improvement=None,  # IMP_IMP,
        min_samples_leaf=.001
    )

    jpt_mf.learn(data_moveforward)
    jpt_mf.postprocess_leaves()

    logger.debug(f'...done! saving to file {os.path.join(locs.examples, "robotaction", dt, f"000-MOVEFORWARD.tree")}')

    jpt_mf.save(os.path.join(locs.examples, 'robotaction', dt, f'000-MOVEFORWARD.tree'))

    logger.debug('...done.')


def learn_jpt_turn(dt):
    logger.debug('learning TURN JPT...')

    # learn discriminative JPT from data generated by test_data_curation for TURN
    data_turn = pd.read_csv(
        os.path.join(locs.examples, 'robotaction', dt, 'data', f'000-ALL-TURN.csv'),
        delimiter=',',
        header=0
    )
    turnvars = infer_from_dataframe(data_turn, scale_numeric_types=False)

    jpt_t = JPT(
        variables=turnvars,
        targets=turnvars[3:],
        min_impurity_improvement=IMP_IMP,
        min_samples_leaf=SMPL_LEAF,
        max_depth=5
    )

    jpt_t.learn(data_turn)
    jpt_t.postprocess_leaves()

    logger.debug(f'...done! saving to file {os.path.join(locs.examples, "robotaction", dt, f"000-TURN.tree")}')

    jpt_t.save(os.path.join(locs.examples, 'robotaction', dt, f'000-TURN.tree'))

    logger.debug('...done.')


def plot_jpt_moveforward(dt):
    jpt_mf = JPT.load(os.path.join(locs.examples, 'robotaction', dt, f'000-MOVEFORWARD.tree'))
    logger.debug('plotting MOVEFORWARD tree without distribution plots...')
    jpt_mf.plot(
        title='MOVEFORWARD',
        filename=f'000-MOVEFORWARD-nodist',
        directory=os.path.join(locs.examples, 'robotaction', dt, 'plots'),
        leaffill='#CCDAFF',
        nodefill='#768ABE',
        alphabet=True,
        view=SHOWPLOTS
    )

    logger.debug('plotting MOVEFORWARD tree...')
    jpt_mf.plot(
        title='MOVEFORWARD',
        plotvars=list(jpt_mf.variables),
        filename=f'000-MOVEFORWARD',
        directory=os.path.join(locs.examples, 'robotaction', dt, 'plots'),
        leaffill='#CCDAFF',
        nodefill='#768ABE',
        alphabet=True,
        view=SHOWPLOTS
    )


def plot_jpt_turn(dt):
    jpt_t = JPT.load(os.path.join(locs.examples, 'robotaction', dt, f'000-TURN.tree'))
    logger.debug('plotting TURN tree...')
    jpt_t.plot(
        title='TURN',
        plotvars=list(jpt_t.variables),
        filename=f'000-TURN',
        directory=os.path.join(locs.examples, 'robotaction', dt, 'plots'),
        leaffill='#CCDAFF',
        nodefill='#768ABE',
        alphabet=True,
        view=SHOWPLOTS
    )

# params
RUNS = 1000
NUMACTIONS = 100
IMP_IMP = 0
SMPL_LEAF = 1
COLLIDED = True  # use collided (symbolic) variable
SEMI = True  # semi = True: randomly select position and move around in circles, semi = False: generate paths
USEDELTAS = True
USE_RECENT = True
SHOWPLOTS = False
LEARNONLY = True


if __name__ == '__main__':
    init_loggers(level='debug')

    # use most recently created dataset or create from scratch
    if USE_RECENT:
        DT = recent_example(os.path.join(locs.examples, 'robotaction'))
        DT = os.path.join(locs.examples, 'robotaction', '2023-08-02_14:23')
    else:
        DT = f'{datetime.datetime.now().strftime(FILESTRFMT)}'

    logger.debug(f'running robotaction data generation with timestamp {DT}')

    if not os.path.exists(os.path.join(locs.examples, 'robotaction', DT)):
        os.mkdir(os.path.join(locs.examples, 'robotaction', DT))
        os.mkdir(os.path.join(locs.examples, 'robotaction', DT, 'plots'))
        os.mkdir(os.path.join(locs.examples, 'robotaction', DT, 'data'))

    if not LEARNONLY:
        if SEMI:
            robot_pos_semi_random(DT)
            # data_curation_semi(DT)
        else:
            robot_pos_random(DT)
            data_curation(DT, usedeltas=USEDELTAS)

    learn_jpt_moveforward(DT)
    # learn_jpt_turn(DT)

    plot_jpt_moveforward(DT)
    # plot_jpt_turn(DT)
