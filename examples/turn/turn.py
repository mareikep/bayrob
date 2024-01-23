import datetime
import os

import dnutils
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from calo.models.action import Move
from calo.models.world import GridAgent, Grid
from calo.utils import locs
from calo.utils.constants import FILESTRFMT, calologger
from calo.utils.dynamic_array import DynamicArray
from calo.utils.plotlib import defaultconfig
from calo.utils.utils import recent_example

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


def generate_data(fp, args):

    lrturns = args.lrturns if 'lrturns' in args else 360
    range_t = args.range_t if 'range_t' in args else 45

    logger.debug(f'Generating {lrturns * 2 * range_t} direction data points...')

    # init agent at left lower corner facing right
    a = GridAgent(
        world=w
    )

    # set initial position and facing direction
    a.pos = (0, 0)
    a.dir = (1, 0)
    initdir = a.dir

    dt_ = DynamicArray(shape=(lrturns * 2 * range_t, 5), dtype=np.float32)

    for degi in np.random.uniform(low=-180, high=180, size=lrturns):

        # turn to new starting direction
        Move.turndeg(a, degi)
        curdir = a.dir

        # make additional turns uniformly distributed to the left and right
        # in a -x/+x degree range
        for randdeg in np.random.uniform(low=-range_t, high=range_t, size=range_t*2):
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
        'angle': np.float32,
        'xdir_out': np.float32,
        'ydir_out': np.float32,
    })

    logger.debug(f"...done! Saving {data_turn.shape[0]} data points to {os.path.join(fp, 'data', f'000-{args.example}.parquet')}...")
    data_turn.to_parquet(os.path.join(fp, 'data', f'000-{args.example}.parquet'), index=False)

    return data_turn


def plot_data(fp, args) -> go.Figure:
    logger.debug('plotting data...')

    df = pd.read_parquet(os.path.join(fp, 'data', f'000-{args.example}.parquet'))

    fig_d = px.scatter(
        df,
        x="xdir_in",
        y="ydir_in",
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
        os.path.join(fp, 'plots', f'000-{args.example}-data.html'),
        config=defaultconfig,
        include_plotlyjs="cdn"
    )

    fig_d.to_json(os.path.join(fp, 'plots', f'000-{args.example}-data.json'))
    fig_d.write_image(os.path.join(fp, 'plots', f'000-{args.example}-data.png'))
    fig_d.write_image(os.path.join(fp, 'plots', f'000-{args.example}-data.svg'))

    fig_d.show(config=defaultconfig)

    return fig_d


# init agent and world
w = Grid(
    x=[0, 100],
    y=[0, 100]
)


def init(fp, args):
    logger.debug('Initializing obstacles...')

    if args.obstacles:
        obstacles = [
            ((15, 10, 25, 20), "chair1"),
            ((35, 10, 45, 20), "chair2"),
            ((10, 30, 50, 50), "kitchen_island"),
            ((80, 30, 100, 70), "stove"),
            ((10, 80, 50, 100), "kitchen_unit"),
            ((60, 80, 80, 100), "fridge"),
        ]

        for o, n in obstacles:
            w.obstacle(*o, name=n)


def learn_jpt(
        fp,
        args
):
    raise NotImplementedError


def teardown(fp, args):
    pass


def main():
    from argparse import Namespace

    args = Namespace(
        recent=False,
        data=False
    )

    if args.recent:
        DT = recent_example(os.path.join(locs.examples, 'robotaction'))
        logger.debug(f'Using recent directory {DT}')
    else:
        DT = f'{datetime.datetime.now().strftime(FILESTRFMT)}'
        logger.debug(f'Creating new directory {DT}')

    fp = os.path.join(locs.examples, 'perception', DT)

    if not os.path.exists(fp):
        logger.debug(f'creating directory {fp}')
        os.mkdir(fp)
        os.mkdir(os.path.join(fp, 'plots'))
        os.mkdir(os.path.join(fp, 'data'))

    if not args.recent:
        generate_data(fp, args)

    if args.learn:
        from examples.examples import learn_jpt
        learn_jpt(fp, args)

    if args.plot:
        from examples.examples import plot_jpt
        plot_jpt(fp, args)

    if args.data:
        plot_data(fp, args)
