import datetime
import math
import os
from itertools import product
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
from calo.utils.plotlib import defaultconfig, plotly_sq, plot_data_subset, to_rgb
from calo.utils.utils import recent_example
from jpt import infer_from_dataframe, JPT
from jpt.base.intervals import ContinuousSet
from jpt.distributions import Gaussian

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


def robot_dir_data(fp, lrturns=500):
    logger.debug('Generating direction data...')

    # init agent at left lower corner facing right
    a = GridAgent(
        world=w
    )

    # set initial position and facing direction
    a.pos = (0, 0)
    a.dir = (1, 0)
    initdir = a.dir

    dt_ = DynamicArray(shape=(lrturns*100, 5), dtype=np.float32)

    for degi in np.random.uniform(low=-180, high=180, size=lrturns):

        # turn to new starting direction
        Move.turndeg(a, degi)
        curdir = a.dir

        # make 30 additional turns uniformly distributed to the left and right
        # in a -20/+20 degree range
        for randdeg in np.random.uniform(low=-25, high=25, size=100):
            # turn and save new direction
            Move.turndeg(a, randdeg)
            dt_.append(np.array(
                [[
                    *curdir,
                    randdeg,
                    *np.array(a.dir) - np.array(curdir)  # deltas!
                ]])
            )

            a.dir = curdir
        a.dir = initdir

    data_turn = pd.DataFrame(data=dt_.data, columns=['xdir_in', 'ydir_in', 'angle', 'xdir_out', 'ydir_out'])
    data_turn = data_turn.astype({
        'xdir_in': np.float32,
        'ydir_in': np.float32,
        'xdir_out': np.float32,
        'ydir_out': np.float32,
        'angle': np.float32
    })

    logger.debug(f"...done! Saving {data_turn.shape[0]} data points to {os.path.join(fp, 'data', f'000-ALL-TURN.parquet')}...")
    data_turn.to_parquet(os.path.join(fp, 'data', f'000-ALL-TURN.parquet'), index=False)

    return data_turn


def robot_pos_semi_random(fp, limit=100, lrturns=200):
    # for each x/y position in 100x100 grid turn 16 times in positive and negative direction and make one step ahead
    # respectively. check for collision/success
    logger.debug(f'Generating up to {math.pow(limit,2)*lrturns} move data points...')

    # init agent at left lower corner facing right
    a = GridAgent(
        world=w
    )

    dm_ = DynamicArray(shape=(int(math.pow(limit,2)*lrturns), 7), dtype=np.float32)
    for y, x in product(range(0, limit, 1), repeat=2):
        # if the xy pos is inside an obstacle, sampling around it does not make
        # sense, so skip it
        if w.collides([x, y]): continue

        # sample around x/y position to add some gaussian noise
        npos = (Gaussian(x, .3).sample(1), Gaussian(y, .3).sample(1))  # TODO: use noisy position?

        # do not position agent on obstacles (retry 3 times, then skip)
        tries = 0
        while w.collides(npos) and tries <= 5:
            npos = (Gaussian(x, .3).sample(1), Gaussian(y, .3).sample(1))
            tries += 1

        # if the sampled position still collides, skip
        if w.collides(npos): continue

        if x == -limit:
            logger.debug(f'Position : {npos[0]}/{npos[1]}')

        # initially, agent always faces right
        a.pos = npos
        a.dir = initdir = (1., 0.)

        # for each position, uniformly sample lrturns angles from -180 to 180;
        # after each turn, turn again 10 times uniformly distributed in a -20/20 degree range
        # and make one step forward, respectively, save datapoint
        # and step back to initpos (i.e. the sampled pos around x/y)
        # turn to the right
        for degi in np.random.uniform(low=-180, high=180, size=lrturns):

            # turn to new starting direction
            Move.turndeg(a, degi)

            # move forward and save new position/direction
            Move.moveforward(a, 1)
            dm_.append(np.array(
                [[
                    *npos,
                    *a.dir,
                    *np.array(a.pos) - np.array(npos),  # deltas!
                    a.collided
                ]])
            )

            # step back/reset position and direction
            a.dir = initdir
            a.pos = npos

    data_moveforward = pd.DataFrame(data=dm_.data, columns=['x_in', 'y_in', 'xdir_in', 'ydir_in', 'x_out', 'y_out', 'collided'])

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

    logger.debug(f"...done! Saving {data_moveforward.shape[0]} data points to {os.path.join(fp, 'data', f'000-ALL-MOVEFORWARD.parquet')}...")
    data_moveforward.to_parquet(os.path.join(fp, 'data', f'000-ALL-MOVEFORWARD.parquet'), index=False)

    return data_moveforward


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
        targets=movevars[4:],
        min_impurity_improvement=None,
        min_samples_leaf=2000  # .005
    )

    jpt_mf.learn(data_moveforward, close_convex_gaps=False)
    # jpt_mf = jpt_mf.prune(similarity_threshold=.77)
    # jpt_mf.postprocess_leaves()

    logger.debug(f'...done! saving to file {os.path.join(fp, f"000-MOVEFORWARD.tree")}')

    jpt_mf.save(os.path.join(fp, f'000-MOVEFORWARD.tree'))

    logger.debug('...done.')


def learn_jpt_moveforward_constrained(
        fp,
        constraints=None,
        vars=None,
        tgtidx=2,
        name=None
):
    logger.debug('learning constrained MOVEFORWARD JPT...')

    # learn discriminative JPT from data generated by test_data_curation for MOVEFORWARD
    df = pd.read_parquet(
        os.path.join(fp, 'data', f'000-ALL-MOVEFORWARD.parquet'),
    )

    logger.debug('Got dataframe of shape:', df.shape)

    if constraints is not None:
        c = [(var, op, v) for var, val in constraints.items() for v, op in ([(val.lower, ">="), (val.upper, "<=")] if isinstance(val, ContinuousSet) else [(val, "==")])]

        s = ' & '.join([f'({var} {op} {num})' for var, op, num in c])
        logger.debug('Extracting dataset using query: ', s)
        df = df.query(s)
        logger.debug('Returned subset of shape:', df.shape)

    # data_moveforward = data_moveforward[['x_in', 'y_in']]
    df = df[vars]
    movevars = infer_from_dataframe(
        df,
        scale_numeric_types=False
    )

    jpt_mf = JPT(
        variables=movevars,
        targets=movevars[tgtidx:],
        min_impurity_improvement=None,
        min_samples_leaf=2000  # .005
    )

    jpt_mf.learn(df, close_convex_gaps=False)
    # jpt_mf = jpt_mf.prune(similarity_threshold=.77)
    # jpt_mf.postprocess_leaves()

    logger.debug(f'...done! saving to file {os.path.join(fp, name)}')

    jpt_mf.save(os.path.join(fp, name))

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

    logger.debug(f'...done! saving to file {os.path.join(fp, f"000-TURN.tree")}')

    jpt_t.save(os.path.join(fp, f'000-TURN.tree'))

    logger.debug('...done.')


def plot_jpt_moveforward(fp, showplots=False):
    logger.debug('plotting MOVEFORWARD tree without distributions...')
    jpt_mf = JPT.load(os.path.join(fp, f'000-MOVEFORWARD.tree'))
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
    logger.debug('plotting TURN tree without distributions...')
    jpt_t = JPT.load(os.path.join(fp, f'000-TURN.tree'))
    jpt_t.plot(
        title='TURN',
        filename=f'000-TURN-nodist',
        directory=os.path.join(fp, 'plots'),
        leaffill='#CCDAFF',
        nodefill='#768ABE',
        alphabet=True,
        view=showplots
    )

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
    logger.debug('plotting data...')

    df = pd.read_parquet(os.path.join(fp, 'data', f'000-ALL-MOVEFORWARD.parquet'))

    fig_d = px.scatter(
        df,
        x="x_in",
        y="y_in",
        color_continuous_scale=px.colors.sequential.dense,
        color=[1]*df.shape[0],#range(df.shape[0]),
        size=[1]*df.shape[0],
        size_max=5,
        width=1000,
        height=1000
    )

    fig_d.update_layout(
        coloraxis_showscale=False,
    )

    fig_d.write_html(
        os.path.join(fp, 'plots', f'000-DATA-MOVE.html'),
        config=defaultconfig,
        include_plotlyjs="cdn"
    )

    fig_d.to_json(os.path.join(fp, 'plots', f'000-DATA-MOVE.json'))
    fig_d.to_json(os.path.join(fp, 'plots', f'000-DATA-MOVE.png'))
    fig_d.write_image(os.path.join(fp, 'plots', f'000-DATA-MOVE.svg'))

    fig_d.show(config=defaultconfig)

    return fig_d


def plot_world(fp, limit) -> go.Figure:
    logger.debug('plotting world...')

    # plot annotated rectangles representing the obstacles and world boundaries
    fig_o = go.Figure()
    fig_o.update_layout(
        coloraxis_showscale=False,
        width=1000,
        height=1000
    )
    for i, (o, on) in enumerate(zip(w.obstacles, w.obstaclenames)):
        fig_o.add_trace(
            plotly_sq(o, lbl=on if on else f'O{i+1}', color='rgb(15,21,110)', legend=False))

    fig_o.add_trace(
        plotly_sq((limit[0], limit[0], limit[1], limit[1]), lbl="kitchen_boundaries", color='rgb(15,21,110)', legend=False))

    fig_o.write_html(
        os.path.join(fp, 'plots', f'000-MOVE-OBSTACLES.html'),
        config=defaultconfig,
        include_plotlyjs="cdn"
    )

    fig_o.write_image(os.path.join(fp, 'plots', f'000-MOVE-OBSTACLES.png'))
    fig_o.write_image(os.path.join(fp, 'plots', f'000-MOVE-OBSTACLES.svg'))

    fig_o.show(config=defaultconfig)
    return fig_o


def robot_pos_random(fp, runs, nactions):
    logger.debug('generating random robot data...')

    # init agent and world
    w = Grid()
    w.obstacle(25, 25, 50, 50)
    w.obstacle(-10, 10, 0, 40)
    w.obstacle(20, -30, 50, 10)
    w.obstacle(-75, -40, -50, -10)
    w.obstacle(-25, -75, -15, -50)

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


def main(DT, args):

    fp = os.path.join(locs.examples, 'robotaction', DT)

    if not os.path.exists(fp):
        logger.debug(f'creating directory {fp}')
        os.mkdir(fp)
        os.mkdir(os.path.join(fp, 'plots'))
        os.mkdir(os.path.join(fp, 'data'))

    w.obstacle(15, 10, 25, 20, name="chair1")
    w.obstacle(35, 10, 45, 20, name="chair2")
    w.obstacle(10, 30, 50, 50, name="kitchen_island")
    w.obstacle(80, 30, 100, 70, name="stove")
    w.obstacle(10, 80, 50, 100, name="kitchen_unit")
    w.obstacle(60, 80, 80, 100, name="fridge")

    if not args.recent:
        if args.turn:
            robot_dir_data(fp)
        if args.move:
            robot_pos_semi_random(fp)

    if args.turn:
        learn_jpt_turn(fp)
        plot_jpt_turn(fp, args.showplots)

    if args.move:
        learn_jpt_moveforward(fp)
        learn_jpt_moveforward_constrained(
            fp,
            constraints={'collided': True},
            vars=['x_in', 'y_in', 'xdir_in', 'ydir_in', 'x_out', 'y_out'],
            tgtidx=4,
            name="000-MOVEFORWARD-collided.tree"
        )
        learn_jpt_moveforward_constrained(
            fp,
            constraints={'collided': False},
            vars=['xdir_in', 'ydir_in', 'x_out', 'y_out'],
            tgtidx=2,
            name="000-MOVEFORWARD-nocollided.tree"
        )
        plot_jpt_moveforward(fp, args.showplots)

    if args.data:
        plot_world(fp, limit=[0, 100])
        plot_data(fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CALOWeb.')
    parser.add_argument("-v", "--verbose", dest="verbose", default='debug', type=str, action="store", help="Set verbosity level {debug,info,warning,error,critical}. Default is info.")
    parser.add_argument('-r', '--recent', default=False, action='store_true', help='use most recent folder greated', required=False)
    parser.add_argument('-s', '--showplots', default=False, action='store_true', help='show plots', required=False)
    parser.add_argument('-t', '--turn', default=False, action='store_true', help='trigger generating turn data/learning turn model', required=False)
    parser.add_argument('-m', '--move', default=False, action='store_true', help='trigger generating move data/learning move model', required=False)
    parser.add_argument('-d', '--data', default=False, action='store_true', help='trigger generating data/world plots', required=False)
    parser.add_argument('--min-samples-leaf', type=float, default=.01, help='min_samples_leaf parameter', required=False)
    args = parser.parse_args()

    init_loggers(level=args.verbose)

    # use most recently created dataset or create from scratch
    if args.recent:
        DT = recent_example(os.path.join(locs.examples, 'robotaction'))
        logger.debug(f'Using recent directory {DT}')
    else:
        DT = f'{datetime.datetime.now().strftime(FILESTRFMT)}'
        logger.debug(f'Creating new directory {DT}')

    main(DT, args)
