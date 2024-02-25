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
from calo.utils.plotlib import defaultconfig, fig_to_file
from calo.utils.utils import recent_example

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


def generate_data(fp, args):

    datapoints = args.datapoints if 'datapoints' in args else 10000
    range_t = args.range_t if 'range_t' in args else 45

    logger.debug(f'Generating {datapoints} direction data points...')

    # init agent at left lower corner facing right
    a = GridAgent(
        world=w
    )

    # # set initial position and facing direction
    a.pos = (0, 0)
    initdir = (0, 1)
    a.dir = initdir

    dt_ = DynamicArray(shape=(datapoints, 5), dtype=np.float32)
    for deg1, deg2 in np.random.uniform(low=[0, -range_t], high=[360, range_t], size=(datapoints, 2)):

        # turn to new starting direction
        Move.turndeg(a, deg1)
        curdir = a.dir

        # turn and save new direction
        Move.turndeg(a, deg2)
        dt_.append(np.array(
            [[
                *curdir,
                deg2,
                *np.array(a.dir) - np.array(curdir)  # deltas!
            ]])
        )

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

    fig_to_file(fig_d, os.path.join(fp, 'plots', f'000-{args.example}-data.html'), ftypes=['.svg', ".png"])
    fig_d.show(config=defaultconfig(fname=os.path.join(fp, 'plots', f'000-{args.example}-data.html')))

    return fig_d


def crossval(fp, args):
    pass


# init agent and world
w = Grid(
    x=[0, 100],
    y=[0, 100]
)


def init(fp, args):
    pass


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
