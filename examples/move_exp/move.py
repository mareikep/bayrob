import datetime
import os

import dnutils
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from calo.models.action import Move
from calo.models.world import GridAgent, Grid
from calo.utils import locs
from calo.utils.constants import FILESTRFMT, calologger
from calo.utils.dynamic_array import DynamicArray
from calo.utils.plotlib import defaultconfig, plotly_sq, plot_data_subset, fig_to_file
from calo.utils.utils import recent_example

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


def generate_data(fp, args):

    lrturns = args.lrturns if 'lrturns' in args else 200
    areas = args.walkingareas if "areas" in args else False
    dropcollisions = args.dropcollisions if "dropcollisions" in args else False
    keepinsidecollisions = args.insidecollisions if "insidecollisions" in args else True
    logger.error(f'LRTURN: {lrturns}')

    # for each x/y position in 100x100 grid turn 16 times in positive and negative direction and make one step ahead
    # respectively. check for collision/success
    xl, yl, xu, yu = w.coords
    xu = xu
    yu = yu

    # init agent at left lower corner facing right
    a = GridAgent(
        world=w
    )

    if areas:
        # draw samples uniformly from the three walking areas instead of the entire kitchen
        logger.debug(f'Drawing samples from walking areas')
        walking_areas = [
            ((5, 50, 50, 80), "wa1"),
            ((50, 30, 80, 80), "wa2"),
            ((45, 0, 100, 50), "wa3")
        ]
        samplepositions = np.concatenate([np.random.uniform(low=[xl, yl], high=[xu, yu], size=((xu-xl)*(yu-yl), 2)) for (xl, yl, xu, yu), n in walking_areas])
    else:
        # OR draw samples from entire kitchen area
        logger.debug(f'Drawing samples from entire kitchen')
        samplepositions = np.random.uniform(low=[xl, yl], high=[xu, yu], size=((xu-xl)*(yu-yl), 2))

    logger.debug(f'Generating up to {len(samplepositions)*lrturns} move data points...')
    idirs = {0: (1., 0.), 1: (-1., 0.), 2: (0., 1.), 3: (0., -1.)}
    dm_ = DynamicArray(shape=(len(samplepositions), 7), dtype=np.float32)
    for i, (x, y) in enumerate(samplepositions):
        # if the xy pos is inside an obstacle, skip it, otherwise use as sample position
        if w.collides((x, y)):
            if keepinsidecollisions:
                dm_.append(np.array(
                    [[
                        *(x, y),
                        *a.dir,
                        *(0, 0),  # do not move
                        True
                    ]])
                )
            continue

        if i % 100 == 0:
            logger.debug(f"generated {len(dm_)} datapoints for {i} positions so far")

        # initially, agent always faces left, right, up or down
        initdir = idirs[np.random.randint(len(idirs))]
        initpos = (x, y)
        a.dir = initdir
        a.pos = initpos

        # for each position, uniformly sample lrturns angles from -180 to 180;
        # after each turn, make one step forward, save datapoint
        # and step back to initpos (i.e. the sampled pos around x/y)
        # and turn back to initdir
        for degi in np.random.uniform(low=-90, high=91, size=lrturns):

            # turn to new starting direction
            move.turndeg(a, degi)

            # move forward and save new position/direction
            move.moveforward(a, 1)
            dm_.append(np.array(
                [[
                    *initpos,
                    *a.dir,
                    *np.array(a.pos) - np.array(initpos),  # deltas!
                    a.collided
                ]])
            )

            # step back/reset position and direction
            a.dir = initdir
            a.pos = initpos

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

    # remove collision data points
    if dropcollisions:
        logger.debug(f"Dropping collision data points...")
        data_moveforward = data_moveforward[(data_moveforward['x_out'] != 0) | (data_moveforward['y_out'] != 0)]

    logger.debug(f"...done! Saving {data_moveforward.shape[0]} data points to {os.path.join(fp, 'data', f'000-{args.example}.parquet')}...")
    data_moveforward.to_parquet(os.path.join(fp, 'data', f'000-{args.example}.parquet'), index=False)

    return data_moveforward


def plot_data(fp, args) -> go.Figure:
    logger.debug('plotting data...')

    df = pd.read_parquet(os.path.join(fp, 'data', f'000-{args.example}.parquet'))

    fig_d = plot_data_subset(
        df,
        xvar="x_in",
        yvar="y_in",
        constraints={},
        limx=(0, 100),
        limy=(0, 100),
        save=None,
        show=False,
        color='rgb(0,104,180)'
    )
    fig_to_file(fig_d, os.path.join(fp, 'plots', f'000-{args.example}-data.html'), ftypes=['.svg', '.png', '.html'])

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
            plotly_sq(o, lbl=on if on else f'O{i+1}', color='rgb(59, 41, 106)', legend=False))

    fig_o.add_trace(
        plotly_sq(w.coords, lbl="kitchen_boundaries", color='rgb(59, 41, 106)', legend=False))

    fig_to_file(fig_o, os.path.join(fp, 'plots', f'000-{args.example}-obstacles.json'), ftypes=['.svg', '.html', '.png'])

    fig_o.show(config=defaultconfig)
    return fig_o


# init agent and world
w = Grid(
    x=[0, 30],
    y=[0, 30]
)

move = Move(
    # degu=0,
    # distu=0
)


def init(fp, args):
    logger.debug('Initializing obstacles...')

    if args.obstacles:
        obstacles = [
            ((5, 5, 20, 10), "kitchen_island")
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
