import re
import signal
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm

from jpt import JPT
from jpt.distributions import Gaussian
import datetime
import os

import dnutils
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from bayrob.models.action import Move
from bayrob.models.world import GridAgent, Grid
from bayrob.utils import locs
from bayrob.utils.constants import FILESTRFMT, bayroblogger
from bayrob.utils.dynamic_array import DynamicArray
from bayrob.utils.plotlib import defaultconfig, plotly_sq, plot_data_subset, fig_to_file, plot_heatmap
from bayrob.utils.utils import recent_example

logger = dnutils.getlogger(bayroblogger, level=dnutils.DEBUG)

def _initialize_worker_process():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _gendata(data_range):
    a = GridAgent(
        world=w
    )
    initdir = tuple(np.random.uniform(low=[0, 0], high=[1, 1], size=2))
    dm_ = []

    for i, (x, y) in enumerate(data_range):
        # if the xy pos is inside an obstacle, skip it, otherwise use as sample position
        if w.collides((x, y)): continue

        # initially, agent always faces left, right, up or down
        initpos = (x, y)
        a.dir = initdir
        a.pos = initpos

        # for each position, uniformly sample lrturns angles from -180 to 180;
        # after each turn, make one step forward, save datapoint
        # and step back to initpos (i.e. the sampled pos around x/y)
        # and turn back to initdir
        for degi in np.random.uniform(low=0, high=360, size=360):
            ipos = Gaussian(initpos, [[.001, 0], [0, .001]]).sample(1)  # initpos
            a.pos = ipos

            # turn to new starting direction
            move.turndeg(a, degi)

            # move forward and save new position/direction
            move.moveforward(a, 1)
            dm_.append([
                *ipos,
                *a.dir,
                *np.array(a.pos) - np.array(ipos),  # deltas!
                a.collided
            ])

            # step back/reset position and direction
            a.dir = initdir
    return dm_

def generate_data(fp, args):

    lrturns = args.lrturns if 'lrturns' in args else 360
    numpositions = args.numpositions if 'numpositions' in args else None
    areas = args.walkingareas if "areas" in args else False
    dropcollisions = args.dropcollisions if "dropcollisions" in args else False
    dropinnerobstacles = args.dropinnerobstacles if "dropinnerobstacles" in args else False
    keepinsidecollisions = args.insidecollisions if "insidecollisions" in args else False

    # for each x/y position in 100x100 grid turn 16 times in positive and negative direction and make one step ahead
    # respectively. check for collision/success
    xl, yl, xu, yu = w.coords

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
        samplepositions = np.concatenate([np.random.uniform(low=[xl, yl], high=[xu, yu], size=(
        numpositions if numpositions is not None else int((xu - xl) * (yu - yl) * 1.5), 2)) for (xl, yl, xu, yu), n in
                                          walking_areas])
    else:
        # OR draw samples from entire kitchen area
        logger.debug(f'Drawing samples from entire kitchen')
        samplepositions = np.random.uniform(low=[xl, yl], high=[xu, yu], size=(numpositions if numpositions is not None else int((xu - xl) * (yu - yl) * 1.5), 2))

    logger.debug(f'Generating up to {len(samplepositions) * lrturns} move data points representing {len(samplepositions)} positions with {lrturns} turns each...')
    progbar = tqdm(total=len(samplepositions) * lrturns, desc='Generating data points', colour="green")

    dm_ = []  # DynamicArray(shape=(len(samplepositions * lrturns), 7), dtype=np.float32)
    nchunks = 1000
    pool = Pool(
        cpu_count(),
        initializer=_initialize_worker_process
    )
    for i, data_chunk in enumerate(
            pool.imap_unordered(
                _gendata,
                iterable=np.array_split(samplepositions, nchunks)
            )
    ):
        dm_.extend(list(data_chunk))
        # progbar.update(len(data_chunk))
        progbar.update(int(len(samplepositions) / nchunks * lrturns))
    pool.close()
    pool.join()

    progbar.close()

    data_moveforward = pd.DataFrame(data=dm_, columns=['x_in', 'y_in', 'xdir_in', 'ydir_in', 'x_out', 'y_out', 'collided'])

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

    if dropinnerobstacles:
        pattern = 'not ((`x_in` >= {}) & (`x_in` <= {}) & (`y_in` >= {}) & (`y_in` <= {}))'
        q = []
        for o, _ in w.obstacles:
            q.append(pattern.format(o[0], o[2], o[1], o[3]))

        data_moveforward = data_moveforward.query(" & ".join(q))

    logger.debug(f"...done! Saving {data_moveforward.shape[0]} data points to {os.path.join(fp, 'data', f'000-{args.example}.parquet')}...")
    data_moveforward.to_parquet(os.path.join(fp, 'data', f'000-{args.example}.parquet'), index=False)

    return data_moveforward


def plot_data(fp, args) -> go.Figure:
    logger.debug('plotting data...')

    df = pd.read_parquet(os.path.join(fp, 'data', f'000-{args.example}.parquet'))
    xl, yl, xu, yu = w.coords
    fig_d = plot_data_subset(
        df,
        xvar="x_in",
        yvar="y_in",
        constraints={},
        limx=(xl, xu),
        limy=(yl, yu),
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

    fig_to_file(fig_o, os.path.join(fp, 'plots', f'000-{args.example}-obstacles.html'), ftypes=['.svg', '.png'])
    fig_o.show(config=defaultconfig(fname=os.path.join(fp, 'plots', f'000-{args.example}-obstacles.html')))

    return fig_o


def crossval(fp, args):
    d = {}
    for treefile in Path(os.path.join(fp, 'folds')).rglob('*.tree'):
        # load tree file
        tn = treefile.name
        print(f"Loaded tree {tn}")
        t = JPT.load(str(treefile))

        # determine respective test dataset for this tree and load it
        match = re.search(r"(\d+)\.tree", tn)
        if match:
            df_id = match.groups()[0]
        else:
            continue

        print(f"Loading fold {df_id}")
        df_ = pd.read_parquet(os.path.join(fp, 'folds', f'000-{args.example}-fold-{df_id}.parquet'))

        # for each datapoint in test dataset, calculate and save likelihood
        print(f"Calculating likelihoods")
        probs, probspervar = t.likelihood(df_, single_likelihoods=True)
        d[tn] = np.mean(probspervar, axis=0)

    data = list(d.values())
    data = pd.DataFrame(
        data=[[np.array(list(d.keys())), np.array([v.name for v in t.variables]), np.array([np.around(d, decimals=2) for d in data]).T, np.array(data).T, np.array(data).T]],
        columns=['tree', 'variable', 'text', 'z', 'lbl']
    )

    # draw matrix tree x datapoint = likelihood(tree, datapoint)
    plot_heatmap(
        data=data,
        xvar='tree',
        yvar='variable',
        nolims=True,
        text='text',
        save=os.path.join(fp, 'folds', 'test.html')
    )


# init agent and world
w = Grid(
    x=[0, 100],
    y=[0, 100]
)

move = Move(
    # degu=0,
    # distu=0
)


def init(fp, args):
    logger.debug('Initializing obstacles...')

    if 'mini' in args and args.mini:
        w.coords = [0, 0, 30, 30]

    if args.obstacles:
        obstacles = [
            # ((5, 5, 20, 10), "kitchen_island"),
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
