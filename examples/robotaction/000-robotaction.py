import datetime
import os
from random import randint

import argparse
import dnutils
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt, patches

from calo.logs.logs import init_loggers
from calo.models.action import Move
from calo.models.world import GridAgent, Grid
from calo.utils import locs
from calo.utils.constants import FILESTRFMT, calologger
from calo.utils.dynamic_array import DynamicArray
from calo.utils.plotlib import defaultconfig, plotly_sq
from calo.utils.utils import recent_example
from jpt import infer_from_dataframe, JPT
from jpt.distributions import Gaussian

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


def robot_pos_semi_random(fp, limit=100, lrturns=20):
    # for each x/y position in 100x100 grid turn 16 times in positive and negative direction and make one step ahead
    # respectively. check for collision/success
    logger.debug('Generating star-shaped robot data...')

    # init agent at left lower corner facing right
    a = GridAgent(
        world=w
    )
    deg_i = 360/lrturns  # intended degree of turn
    g_deg_i = Gaussian(deg_i, 5)

    # write sample data for MOVEFORWARD and TURN action of robot (absolute positions)
    cnt = 0

    dm_ = DynamicArray(shape=(int(7e6), 7), dtype=np.float32)
    dt_ = DynamicArray(shape=(int(7e6), 5), dtype=np.float32)
    for y in range(-limit, limit, 1):

        for x in range(-limit, limit, 1):
            if x == -limit:
                print(f'x/y: {x}/{y}')

            # sample around x/y position to add some gaussian noise
            npos = (Gaussian(x, .3).sample(1), Gaussian(y, .3).sample(1))

            # do not position agent on obstacles
            if w.collides(npos):
                continue

            a.pos = npos
            # initially, agent always faces right
            a.dir = (1., 0.)
            initpos = a.pos
            idir = a.dir

            # for each position, turn lrturns times in positive and negative direction;
            # after each turn, turn again 30 times uniformly distributed in a -20/20 degree range
            # and make one step forward, respectively, save datapoint
            # and step back to initpos (i.e. the sampled pos around x/y)

            # turn to the right
            for _ in range(lrturns):

                degi = g_deg_i.sample(1)
                initdir = a.dir  # save dir before update

                # turn and save new position/direction
                Move.turndeg(a, degi)
                dt_.append(np.array(
                    [
                        *initdir,
                        degi,
                        *np.array(a.dir) - np.array(initdir)  # deltas!
                    ])
                )

                curdir = a.dir

                # make 30 additional turns uniformly distributed to the left and right
                # in a -20/+20 degree range
                for randdeg in np.random.uniform(low=-20, high=20, size=10):
                    # turn and save new position/direction
                    Move.turndeg(a, randdeg)
                    dt_.append(np.array(
                        [
                            *curdir,
                            randdeg,
                            *np.array(a.dir) - np.array(curdir)  # deltas!
                        ])
                    )

                    # move forward and save new position/direction
                    Move.moveforward(a, 1)
                    dm_.append(np.array(
                        [
                            *initpos,
                            *a.dir,
                            *np.array(a.pos) - np.array(initpos),  # deltas!
                            a.collided
                        ])
                    )

                    # step back/reset position and direction
                    a.dir = curdir
                    a.pos = initpos

            a.dir = idir

            # turn to the left
            for _ in range(lrturns):

                degi = -g_deg_i.sample(1)
                initdir = a.dir

                Move.turndeg(a, degi)
                dt_.append(np.array(
                    [
                        *initdir,
                        degi,
                        *np.array(a.dir) - np.array(initdir)  # deltas!
                    ])
                )

                curdir = a.dir

                # make 30 additional turns uniformly distributed to the left and right
                # in a -20/+20 degree range
                for randdeg in np.random.uniform(low=-20, high=20, size=10):
                    Move.turndeg(a, randdeg)
                    dt_.append(np.array(
                        [
                            *curdir,
                            randdeg,
                            *np.array(a.dir) - np.array(curdir)  # deltas!
                        ])
                    )

                    Move.moveforward(a, 1)

                    dm_.append(np.array(
                        [
                            *initpos,
                            *a.dir,
                            *np.array(a.pos) - np.array(initpos),  # deltas!
                            a.collided
                        ])
                    )

                    # step back/reset position and direction
                    a.dir = curdir
                    a.pos = initpos

    data_moveforward = pd.DataFrame(data=dm_.data, columns=['x_in', 'y_in', 'xdir_in', 'ydir_in', 'x_out', 'y_out', 'collided'])
    data_turn = pd.DataFrame(data=dt_.data, columns=['xdir_in', 'ydir_in', 'angle', 'xdir_out', 'ydir_out'])

    # save data
    data_moveforward = data_moveforward.astype({
        'x_in': np.float32,
        'y_in': np.float32,
        'xdir_in': np.float32,
        'ydir_in': np.float32,
        'x_out': np.float32,
        'y_out': np.float32,
        'collided': bool
    })
    data_moveforward.to_parquet(os.path.join(fp, 'data', f'000-ALL-MOVEFORWARD.parquet'), index=False)

    data_turn = data_turn.astype({
        'xdir_in': np.float32,
        'ydir_in': np.float32,
        'xdir_out': np.float32,
        'ydir_out': np.float32,
        'angle': np.float32
    })
    data_turn.to_parquet(os.path.join(fp, 'data', f'000-ALL-TURN.parquet'), index=False)

    logger.debug(f'...done! Saving data to {fp}...')

    return data_moveforward, data_turn


def learn_jpt_moveforward(fp):
    logger.debug('learning MOVEFORWARD JPT...')

    # learn discriminative JPT from data generated by test_data_curation for MOVEFORWARD
    data_moveforward = pd.read_parquet(
        os.path.join(fp, 'data', f'000-ALL-MOVEFORWARD.parquet'),
    )

    # data_moveforward = data_moveforward[['x_in', 'y_in']]
    # data_moveforward = data_moveforward[['x_in', 'y_in', 'x_out', 'y_out']]
    movevars = infer_from_dataframe(
        data_moveforward,
        scale_numeric_types=False
    )

    jpt_mf = JPT(
        variables=movevars,
        # targets=movevars[4:],
        min_impurity_improvement=None,
        min_samples_leaf=2000#.005
    )

    jpt_mf.learn(data_moveforward, close_convex_gaps=False)
    # jpt_mf = jpt_mf.prune(similarity_threshold=.77)
    # jpt_mf.postprocess_leaves()

    logger.debug(f'...done! saving to file {os.path.join(fp, f"000-MOVEFORWARD.tree")}')

    jpt_mf.save(os.path.join(fp, f'000-MOVEFORWARD.tree'))

    logger.debug('...done.')


def learn_jpt_turn(fp):
    logger.debug('learning TURN JPT...')

    # learn discriminative JPT from data generated by test_data_curation for TURN
    data_turn = pd.read_parquet(
        os.path.join(fp, 'data', f'000-ALL-TURN.parquet'),
    )

    turnvars = infer_from_dataframe(data_turn, scale_numeric_types=False)

    jpt_t = JPT(
        variables=turnvars,
        targets=turnvars[3:],
        min_impurity_improvement=0,
        min_samples_leaf=1,
        max_depth=5
    )

    jpt_t.learn(data_turn)
    jpt_t.postprocess_leaves()

    logger.debug(f'...done! saving to file {os.path.join(fp, f"000-TURN.tree")}')

    jpt_t.save(os.path.join(fp, f'000-TURN.tree'))

    logger.debug('...done.')


def plot_jpt_moveforward(fp, showplots=False):
    jpt_mf = JPT.load(os.path.join(fp, f'000-MOVEFORWARD.tree'))
    logger.debug('plotting MOVEFORWARD tree without distribution plots...')
    jpt_mf.plot(
        title='MOVEFORWARD',
        filename=f'000-MOVEFORWARD-nodist',
        directory=os.path.join(fp, 'plots'),
        leaffill='#CCDAFF',
        nodefill='#768ABE',
        alphabet=True,
        view=showplots
    )

    logger.debug('plotting MOVEFORWARD tree...')
    jpt_mf.plot(
        title='MOVEFORWARD',
        plotvars=list(jpt_mf.variables),
        filename=f'000-MOVEFORWARD',
        directory=os.path.join(fp, 'plots'),
        leaffill='#CCDAFF',
        nodefill='#768ABE',
        alphabet=True,
        view=showplots
    )


def plot_jpt_turn(fp, showplots=False):
    jpt_t = JPT.load(os.path.join(fp, f'000-TURN.tree'))
    logger.debug('plotting TURN tree...')
    jpt_t.plot(
        title='TURN',
        plotvars=list(jpt_t.variables),
        filename=f'000-TURN',
        directory=os.path.join(fp, 'plots'),
        leaffill='#CCDAFF',
        nodefill='#768ABE',
        alphabet=True,
        view=showplots
    )


def plot_data(fp) -> go.Figure:

    df = pd.read_parquet(os.path.join(fp, 'data', f'000-ALL-MOVEFORWARD.parquet'))

    fig_s = px.scatter(
        df,
        x="x_in",
        y="y_in",
        color_continuous_scale=px.colors.sequential.dense,
        color=[1]*df.shape[0],#range(df.shape[0]),
        size=[1]*df.shape[0],
        size_max=5,
        width=1700,
        height=1000
    )

    fig_s.update_layout(coloraxis_showscale=False)

    fig_s.write_html(
        os.path.join(fp, 'plots', f'000-TRAJECTORIES-MOVE.html'),
        config=defaultconfig,
        include_plotlyjs="cdn"
    )

    fig_s.to_json(os.path.join(fp, 'plots', f'000-TRAJECTORIES-MOVE.json'))

    # plot annotated rectangles representing the obstacles
    for i, o in enumerate(w.obstacles):
        fig_s.add_trace(
            plotly_sq(o, lbl=f'O{i+1}', color="60,60,60", legend=False))

    fig_s.write_html(
        os.path.join(fp, 'plots', f'000-TRAJECTORIES-MOVE_annotated.html'),
        config=defaultconfig,
        include_plotlyjs="cdn"
    )

    fig_s.write_image(os.path.join(fp, 'plots', f'000-TRAJECTORIES-MOVE_annotated.png'))

    fig_s.show(config=defaultconfig)
    return fig_s


def robot_pos_random(fp, runs, nactions):
    logger.debug('Generating random robot data...')

    # init agent and world
    w = Grid()
    w.obstacle(25, 25, 50, 50)
    w.obstacle(-10, 10, 0, 40)
    w.obstacle(20, -30, 50, 10)
    w.obstacle(-75, -40, -50, -10)
    w.obstacle(-25, -75, -15, -50)

    # set uncertainty
    # Move.DEG_U = .01
    # Move.DIST_U = .01

    gaussian_deg = Gaussian(0, 360)

    fig, ax = plt.subplots(num=1, clear=True)

    # write sample data for MOVEFORWARD and TURN action of robot (absolute positions)
    for j in range(runs):
        poses = []  # for plotting
        turns = []

        # init agent at random position
        a = GridAgent(world=w)
        a.init_random()

        # for each run and action select random
        for _ in range(nactions):
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
        df_moveforward.to_parquet(os.path.join(fp, 'data', f'{j}-MOVEFORWARD.parquet'), index=False)

        df_turn = pd.DataFrame(turns, columns=['xdir', 'ydir', 'angle'])
        df_turn.to_parquet(os.path.join(fp, 'data', f'{j}-TURN.parquet'), index=False)

        # plot trajectories and highlight start and end points
        ax.plot(df_moveforward['x'], df_moveforward['y'], c='cornflowerblue')
        ax.scatter(df_moveforward['x'].iloc[0], df_moveforward['y'].iloc[0], marker='+', c='green', zorder=1000)
        ax.scatter(df_moveforward['x'].iloc[-1], df_moveforward['y'].iloc[-1], marker='+', c='red', zorder=1000)

        # TODO: remove to save storage space and prevent overload of produced images
        plt.savefig(os.path.join(fp, 'plots', f'{j}-MOVE.png'), format="png")

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
    fig.suptitle(f'{runs} runs; {nactions} actions per run')
    fig.canvas.manager.set_window_title(f'000-ALL-MOVE.png')
    plt.legend()
    plt.grid()

    # save and show
    plt.savefig(os.path.join(fp, 'plots', f'000-ALL-MOVE.png'), format="png")

    plt.show()


def data_curation(fp, runs, usedeltas=False):
    logger.debug('curating robot MOVEFORWARD data...')

    # read position data files generated by test_robot_pos and generate large file containing deltas
    # (position-independent)

    data_moveforward = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'x_in', 'y_in', 'x_out', 'y_out', 'collided'])

    for i in range(runs):
        with open(os.path.join(fp, 'data', f'{i}-MOVEFORWARD.parquet'), 'r') as f:
            logger.debug(f"Reading {os.path.join(fp, 'data', f'{i}-MOVEFORWARD.parquet')}...")
            d = pd.read_parquet(f, delimiter=',', header=0)
            for idx, row in d.iterrows():
                if idx == d.index.max(): break

                data_moveforward.loc[idx + i * runs] = [
                    row['xdir'],
                    row['ydir'],
                    row['x'],
                    row['y'],
                    d.iloc[idx + 1]['x']-row['x'] if usedeltas else d.iloc[idx + 1]['x'],
                    d.iloc[idx + 1]['y']-row['y'] if usedeltas else d.iloc[idx + 1]['y'],
                    row['collided']
                ]

    data_moveforward.to_parquet(os.path.join(fp, 'data', f'000-ALL-MOVEFORWARD.parquet'), index=False)

    logger.debug('...done! curating robot TURN data...')
    data_turn = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'angle', 'xdir_out', 'ydir_out'])
    for i in range(runs):
        with open(os.path.join(fp, 'data', f'{i}-TURN.parquet'), 'r') as f:
            logger.debug(f"Reading {os.path.join(fp, 'data', f'{i}-TURN.parquet')}...")
            d = pd.read_parquet(f, delimiter=',', header=0)
            for idx, row in d.iterrows():
                if idx == d.index.max(): continue
                # store data of two consecuting steps
                data_turn.loc[idx + i * runs] = [
                    row['xdir'],
                    row['ydir'],
                    row['angle'],
                    d.iloc[idx + 1]['xdir']-row['xdir'] if usedeltas else d.iloc[idx + 1]['xdir'],
                    d.iloc[idx + 1]['ydir']-row['ydir'] if usedeltas else d.iloc[idx + 1]['ydir']
                ]

    data_turn.to_parquet(os.path.join(fp, 'data', f'000-ALL-TURN.parquet'), index=False)
    logger.debug('...done!')


# init agent and world
w = Grid()


def main(DT, recent=False, showplots=False):

    fp = os.path.join(locs.examples, 'robotaction', DT)

    if not os.path.exists(fp):
        logger.debug(f'creating directory {fp}')
        os.mkdir(fp)
        os.mkdir(os.path.join(fp, 'plots'))
        os.mkdir(os.path.join(fp, 'data'))

    w.obstacle(25, 25, 50, 50)
    w.obstacle(-10, 10, 0, 40)
    w.obstacle(50, -30, 20, 10)
    w.obstacle(-75, -10, -50, -40)
    w.obstacle(-25, -50, -15, -75)

    if not recent:
        robot_pos_semi_random(fp)

    learn_jpt_moveforward(fp)
    # learn_jpt_turn(fp)

    plot_jpt_moveforward(fp, showplots)
    # plot_jpt_turn(fp, showplots)
    # plot_data(fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CALOWeb.')
    parser.add_argument("-v", "--verbose", dest="verbose", default='info', type=str, action="store", help="Set verbosity level {debug,info,warning,error,critical}. Default is info.")
    parser.add_argument('-r', '--recent', default=False, action='store_true', help='use most recent folder greated', required=False)
    parser.add_argument('-s', '--showplots', default=False, action='store_true', help='show plots', required=False)
    parser.add_argument('--min-samples-leaf', type=float, default=.01, help='min_samples_leaf parameter', required=False)
    args = parser.parse_args()

    init_loggers(level='debug')

    # use most recently created dataset or create from scratch
    if args.recent:
        DT = recent_example(os.path.join(locs.examples, 'robotaction'))
        logger.debug(f'Using recent directory {DT}')
    else:
        DT = f'{datetime.datetime.now().strftime(FILESTRFMT)}'
        logger.debug(f'Creating new directory {DT}')

    main(DT, recent=args.recent, showplots=args.showplots)
