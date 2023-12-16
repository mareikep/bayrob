import datetime
import os
from functools import reduce

import dnutils
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from calo.models.action import Move
from calo.models.world import GridAgent, Grid
from calo.utils import locs
from calo.utils.constants import calologger, FILESTRFMT
from calo.utils.plotlib import defaultconfig
from calo.utils.utils import recent_example
from jpt.distributions import Gaussian

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


def circumference(o):
    return sum([2*edge for edge in [abs(o[2]-o[0]), abs(o[3]-o[1])]])

def generate_gaussian_samples(gaussians, n):
    per_gaussian = int(n / len(gaussians))
    data = [g.sample(per_gaussian) for g in gaussians]
    colors = [[c] * per_gaussian for c in ['#ff0000', '#0000ff', '#00ff00', '#0f0f0f', '#f0f0f0'][:len(gaussians)]]

    all_data = np.vstack(data)

    df = pd.DataFrame({'X': all_data[:, 0], 'Y': all_data[:, 1], 'Color': reduce(list.__add__, colors)})
    return df


def generate_data(fp, args):
    name = args.example

    # the number of samples to generate per point along an obstacle
    samples_ppnt = args.samples_ppnt if 'samples_ppnt' in args else 5

    # the number of samples to generate for each additional sample position (open_positions)
    samples_ppos = args.samples_ppos if 'samples_ppos' in args else int(samples_ppnt*10)

    cols = {
        'x_in': np.float32,
        'y_in': np.float32,
        'xdir_in': np.float32,
        'ydir_in': np.float32,
        'daytime': str,
        'open(fridge_door)': bool,
        'open(cupboard_door_left)': bool,
        'open(cupboard_door_right)': bool,
        'open(kitchen_unit_drawer)': bool,
        'open(stove_door)': bool,
        'detected(cup)': bool,
        'detected(cutlery)': bool,
        'detected(bowl)': bool,
        'detected(sink)': bool,
        'detected(milk)': bool,
        'detected(beer)': bool,
        'detected(cereal)': bool,
        'detected(stovetop)': bool,
        'detected(pot)': bool,
        'nearest_furniture': str,
    }
    daytimes = ['morning', 'post-breakfast', 'pre-lunchtime', 'lunchtime', 'post-lunchtime', 'pre-dinnertime', 'dinnertime', 'post-dinnertime', 'night']
    dt = dict(zip(range(len(daytimes)), daytimes))

    # define some positions where the robot can see the insides of drawers, cupboards and the fridge to sample from
    open_positions = {
        "stove": [
            (72, 40, [5, .07], [.07, 2], 1, -1, 20, None),
        ],
        "kitchen_unit": [
            (20, 75, [7, .07], [.07, 2], -1, 3, 45, "left"),
            (20, 70, [10, .07], [.07, 1], -1, 3, 45, "drawer"),
            (40, 75, [8, -.07], [-.07, 1], 1, 3, 45, "right"),
        ],
        "fridge": [
            (63, 75, [5, -.07], [-.07, 1], 1, 1, 30, None)
        ]
    }

    numdata = int(sum([circumference(o) for o in w.obstacles])*samples_ppnt) + int(sum([len(l) for l in open_positions.values()])*samples_ppos)
    logger.debug(f'Generating up to {numdata} move data points...')

    # init agent at left lower corner facing right
    a = GridAgent(
        world=w
    )

    distance_to_obstacle = 4

    dp_ = []
    # sample 500 positions around each obstacle (all doors/cabinets/drawers closed)
    for o, obstacle in zip(w.obstacles, w.obstaclenames):
        logger.debug(f"sampling data for obstacle {obstacle}")
        # sample positions and directions along edges of obstacles
        for edge in ['upper', 'lower', 'left', 'right']:
            logger.debug(f"edge {edge}")

            if edge == 'upper':
                mudx = 0
                mudy = -1
                dx = np.random.uniform(low=o[0], high=o[2], size=int(abs(o[2]-o[0])*samples_ppnt))
                dy = Gaussian(o[3] + distance_to_obstacle/2, distance_to_obstacle/2).sample(int(abs(o[2]-o[0])*samples_ppnt))
            elif edge == 'lower':
                mudx = 0
                mudy = 1
                dx = np.random.uniform(low=o[0], high=o[2], size=int(abs(o[2]-o[0])*samples_ppnt))
                dy = Gaussian(o[1] - distance_to_obstacle/2, distance_to_obstacle/2).sample(int(abs(o[2]-o[0])*samples_ppnt))
            elif edge == 'left':
                mudx = 1
                mudy = 0
                dx = Gaussian(o[0] - distance_to_obstacle/2, distance_to_obstacle/2).sample(int(abs(o[3]-o[1])*samples_ppnt))
                dy = np.random.uniform(low=o[1], high=o[3], size=int(abs(o[3]-o[1])*samples_ppnt))
            else:
                mudx = -1
                mudy = 0
                dx = Gaussian(o[2] + distance_to_obstacle/2, distance_to_obstacle/2).sample(int(abs(o[3]-o[1])*samples_ppnt))
                dy = np.random.uniform(low=o[1], high=o[3], size=int(abs(o[3]-o[1])*samples_ppnt))

            deg = Gaussian(0, 45).sample(len(dx))
            ddx = ddy = []
            for d in deg:
                a.dir = (mudx, mudy)
                Move.turndeg(a, d)
                ddx_, ddy_ = a.dir
                ddx.append(ddx_)
                ddy.append(ddy_)

            daytime = np.random.randint(low=0, high=len(daytimes), size=len(dx))

            for x_, y_, dx_, dy_, daytime in zip(dx, dy, ddx, ddy, daytime):
                if w.collides((x_, y_)): continue

                npos = (x_, y_)
                ndir = (dx_, dy_)
                daytime = dt[daytime]

                dp_.append(
                    [
                        *npos,  # position
                        *ndir,  # direction
                        daytime,  # time of day
                        False,  # open(fridge_door)
                        False,  # open(cupboard_door_left)
                        False,  # open(cupboard_door_right)
                        False,  # open(kitchen_unit_drawer)
                        False,  # open(stove_door)
                        obstacle == "kitchen_island" and daytime == "morning" or obstacle == "kitchen_unit" and daytime == "post-breakfast",  # see cup
                        obstacle == "kitchen_unit" and daytime == "post-breakfast",  # see cutlery
                        obstacle == "kitchen_island" and daytime == "morning" or obstacle == "kitchen_unit" and daytime == "post-breakfast",  # see bowl
                        obstacle == "kitchen_unit",  # see sink
                        obstacle == "kitchen_island" and daytime == "morning",  # see milk
                        (obstacle == 'kitchen_island' and daytime == "night") or obstacle == "stove",  # see beer
                        obstacle == "kitchen_island" and daytime == "morning",  # see cereal
                        obstacle == 'stove',  # see stovetop
                        (obstacle == 'stove' and daytime in ['pre-lunchtime', 'pre-dinnertime']) or (obstacle == "kitchen_island" and daytime in ['lunchtime', 'dinnertime']) or (obstacle == "kitchen_unit" and daytime in ["post-lunchtime", "post-dinnertime"]),  # see pot
                        obstacle,
                    ]
                )

    # sample another 500 positions near certain furniture objects for generating data points
    # where the respective door is open
    for obstacle, queries in open_positions.items():
        for mx_, my_, stdx_, stdy_, dx_, dy_, deg, side in queries:
            gausspos = Gaussian([mx_, my_], [stdx_, stdy_])

            df = generate_gaussian_samples([gausspos], samples_ppos*2)

            dx = df['X'].values
            dy = df['Y'].values

            deg = Gaussian(0, deg).sample(samples_ppos)
            ddx = ddy = []
            for d in deg:
                a.dir = (dx_, dy_)
                Move.turndeg(a, d)
                ddx_, ddy_ = a.dir
                ddx.append(ddx_)
                ddy.append(ddy_)

            daytime = np.random.randint(low=0, high=len(daytimes), size=samples_ppos)
            for x_, y_, dx_, dy_, daytime in zip(dx, dy, ddx, ddy, daytime):
                if w.collides((x_, y_)): continue

                open_fridge = obstacle == "fridge" or bool(np.random.randint(low=0, high=2))
                open_kitchenunit_left = obstacle == "kitchen_unit" and side == "left" or bool(np.random.randint(low=0, high=2))
                open_kitchenunit_right = obstacle == "kitchen_unit" and side == "right" or bool(np.random.randint(low=0, high=2))
                open_kitchenunit_drawer = obstacle == "kitchen_unit" and side == "drawer" or bool(np.random.randint(low=0, high=2))
                open_stove = obstacle == "stove" or bool(np.random.randint(low=0, high=2))

                npos = (x_, y_)
                ndir = (dx_, dy_)
                daytime = dt[daytime]
                dp_.append(
                    [
                        *npos,  # position
                        *ndir,  # direction
                        daytime,  # time of day
                        open_fridge,  # open(fridge_door)
                        open_kitchenunit_left,  # open(cupboard_door_left)
                        open_kitchenunit_right,  # open(cupboard_door_right)
                        open_kitchenunit_drawer,  # open(kitchen_unit_drawer)
                        open_stove,  # open(stove_door)
                        obstacle == "kitchen_unit" and open_kitchenunit_left and daytime != "morning",  # see cup
                        obstacle == "kitchen_unit" and open_kitchenunit_drawer and daytime != "morning",  # see cutlery
                        obstacle == "kitchen_unit" and open_kitchenunit_left and daytime != "morning",  # see bowl
                        obstacle == "kitchen_unit",  # see sink
                        obstacle == "fridge" and open_fridge and daytime != "morning",  # see milk
                        obstacle == "fridge" and open_fridge or obstacle == 'stove',  # see beer
                        obstacle == "kitchen_unit" and open_kitchenunit_right and daytime != "morning",  # see cereal
                        obstacle == 'stove',  # see stovetop
                        obstacle == "stove" and open_stove and daytime not in ['lunchtime', 'dinnertime'] or obstacle == "kitchen_unit" and daytime in ['post-lunchtime', 'post-dinnertime'],  # see pot
                        obstacle,
                    ]
                )

    df = pd.DataFrame(
        data=dp_,
        columns=list(cols.keys())
    )

    # set type for data columns
    df = df.astype(cols)

    # save data
    logger.debug(f"...done! Saving {df.shape[0]} data points to {os.path.join(fp, 'data', f'000-{name}.parquet')}...")
    df.to_parquet(os.path.join(fp, 'data', f'000-{name}.parquet'), index=False)

    return df


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
    fig_d.to_json(os.path.join(fp, 'plots', f'000-{args.example}-data.png'))
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
        data=False)

    if args.recent:
        DT = recent_example(os.path.join(locs.examples, 'perception'))
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
