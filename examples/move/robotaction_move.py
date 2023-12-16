import datetime
import os
from itertools import product

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
from calo.utils.plotlib import defaultconfig, plotly_sq
from calo.utils.utils import recent_example
from jpt.distributions import Gaussian

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


def generate_data(fp, args):

    lrturns = args.lrturns if 'lrturns' in args else 200

    # for each x/y position in 100x100 grid turn 16 times in positive and negative direction and make one step ahead
    # respectively. check for collision/success
    xl, yl, xu, yu = w.coords
    xu = xu+1
    yu = yu+1

    logger.debug(f'Generating up to {((xu-xl)*(yu-yl))*lrturns} move data points...')

    # init agent at left lower corner facing right
    a = GridAgent(
        world=w
    )

    dm_ = DynamicArray(shape=(int(((xu-xl)*(yu-yl))*lrturns), 7), dtype=np.float32)
    for y, x in product(range(xl, xu, 1), repeat=2):
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

        if x == -xl:
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

    logger.debug(f"...done! Saving {data_moveforward.shape[0]} data points to {os.path.join(fp, 'data', f'000-{args.example}.parquet')}...")
    data_moveforward.to_parquet(os.path.join(fp, 'data', f'000-{args.example}.parquet'), index=False)

    return data_moveforward


def plot_data(fp, args) -> go.Figure:
    logger.debug('plotting data...')

    df = pd.read_parquet(os.path.join(fp, 'data', f'000-{args.example}.parquet'))

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
        os.path.join(fp, 'plots', f'000-{args.example}-data.html'),
        config=defaultconfig,
        include_plotlyjs="cdn"
    )
    
    fig_d.to_json(os.path.join(fp, 'plots', f'000-{args.example}-data.json'))
    fig_d.write_image(os.path.join(fp, 'plots', f'000-{args.example}-data.png'))
    fig_d.write_image(os.path.join(fp, 'plots', f'000-{args.example}-data.svg'))

    fig_d.show(config=defaultconfig)

    plot_world(fp, args)

    return fig_d


def plot_world(fp, args) -> go.Figure:
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
        plotly_sq(w.coords, lbl="kitchen_boundaries", color='rgb(15,21,110)', legend=False))

    fig_o.write_html(
        os.path.join(fp, 'plots', f'000-{args.example}-obstacles.html'),
        config=defaultconfig,
        include_plotlyjs="cdn"
    )

    fig_o.to_json(os.path.join(fp, 'plots', f'000-{args.example}-obstacles.json'))
    fig_o.write_image(os.path.join(fp, 'plots', f'000-{args.example}-obstacles.png'))
    fig_o.write_image(os.path.join(fp, 'plots', f'000-{args.example}-obstacles.svg'))

    fig_o.show(config=defaultconfig)
    return fig_o


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

