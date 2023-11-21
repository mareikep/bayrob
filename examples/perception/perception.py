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


def generate_gaussian_samples(gaussians, n):
    per_gaussian = int(n / len(gaussians))
    data = [g.sample(per_gaussian) for g in gaussians]
    colors = [[c] * per_gaussian for c in ['#ff0000', '#0000ff', '#00ff00', '#0f0f0f', '#f0f0f0'][:len(gaussians)]]

    all_data = np.vstack(data)

    df = pd.DataFrame({'X': all_data[:, 0], 'Y': all_data[:, 1], 'Color': reduce(list.__add__, colors)})
    return df


def generate_data(fp, args):
    name = args.example
    samples_per_obstacle = args.samples_per_obstacle if 'samples_per_obstacle' in args else 500

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
        'stove_door_open': bool,
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

    numdata = int(len(w.obstacles)*samples_per_obstacle)
    logger.debug(f'Generating up to {numdata} move data points...')

    # init agent at left lower corner facing right
    a = GridAgent(
        world=w
    )

    distance_to_obstacle = 4

    dp_ = []
    for o, o_name in zip(w.obstacles, w.obstaclenames):
        logger.debug(f"sampling data for obstacle {o_name}")
        # sample positions and directions along edges of obstacles
        for edge in ['upper', 'lower', 'left', 'right']:
            logger.debug(f"edge {edge}")

            if edge == 'upper':
                mudx = 0
                mudy = -1
                dx = np.random.uniform(low=o[0], high=o[2], size=int(samples_per_obstacle / 4))
                dy = Gaussian(o[3], distance_to_obstacle).sample(int(samples_per_obstacle / 4))
            elif edge == 'lower':
                mudx = 0
                mudy = 1
                dx = np.random.uniform(low=o[0], high=o[2], size=int(samples_per_obstacle / 4))
                dy = Gaussian(o[1], distance_to_obstacle).sample(int(samples_per_obstacle / 4))
            elif edge == 'left':
                mudx = 1
                mudy = 0
                dx = Gaussian(o[0], distance_to_obstacle).sample(int(samples_per_obstacle / 4))
                dy = np.random.uniform(low=o[1], high=o[3], size=int(samples_per_obstacle / 4))
            else:
                mudx = -1
                mudy = 0
                dx = Gaussian(o[2], distance_to_obstacle).sample(int(samples_per_obstacle / 4))
                dy = np.random.uniform(low=o[1], high=o[3], size=int(samples_per_obstacle / 4))

            deg = Gaussian(0, 45).sample(int(samples_per_obstacle / 4))
            ddx = ddy = []
            for d in deg:
                a.dir = (mudx, mudy)
                Move.turndeg(a, d)
                ddx_, ddy_ = a.dir
                ddx.append(ddx_)
                ddy.append(ddy_)

            daytime = np.random.randint(low=0, high=4, size=int(samples_per_obstacle/4))

            for x_, y_, dx_, dy_, tod in zip(dx, dy, ddx, ddy, daytime):
                if w.collides((x_, y_)): continue

                npos = (x_, y_)
                ndir = (dx_, dy_)
                tod = {0: "morning", 1: "day", 2: "evening", 3: "night"}[tod]

                dp_.append(
                    [
                        *npos,  # position
                        *ndir,  # direction
                        tod,  # time of day
                        False,  # open(fridge_door)
                        False,  # open(cupboard_door_left)
                        False,  # open(cupboard_door_right)
                        False,  # open(kitchen_unit_drawer)
                        False,  # stove_door_open
                        o_name == "kitchen_island" and tod == "morning",  # see cup
                        False,  # see cutlery
                        o_name == "kitchen_island" and tod == "morning",  # see bowl
                        o_name == "kitchen_unit",  # see sink
                        o_name == "kitchen_island" and tod == "morning",  # see milk
                        (o_name == 'kitchen_island' and tod == "night") or o_name == "stove",  # see beer
                        o_name == "kitchen_island" and tod == "morning",  # see cereal
                        o_name == 'stove',  # see stovetop
                        (o_name == 'stove' and tod not in ['evening', 'day']) or (o_name == "kitchen_island" and tod in ['evening', 'day']),  # see pot
                        o_name,
                    ]
                )

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


        for mx_, my_, stdx_, stdy_, dx_, dy_, deg, side in open_positions.get(o_name, []):
            gausspos = Gaussian([mx_, my_], [stdx_, stdy_])

            df = generate_gaussian_samples([gausspos], 1000)

            dx = df['X'].values
            dy = df['Y'].values

            deg = Gaussian(0, deg).sample(int(samples_per_obstacle / 4))
            ddx = ddy = []
            for d in deg:
                a.dir = (mudx, mudy)
                Move.turndeg(a, d)
                ddx_, ddy_ = a.dir
                ddx.append(ddx_)
                ddy.append(ddy_)

            daytime = np.random.randint(low=0, high=4, size=int(samples_per_obstacle / 4))
            for x_, y_, dx_, dy_, tod in zip(dx, dy, ddx, ddy, daytime):
                if w.collides((x_, y_)): continue

                npos = (x_, y_)
                ndir = (dx_, dy_)
                tod = {0: "morning", 1: "day", 2: "evening", 3: "night"}[tod]
                dp_.append(
                    [
                        *npos,  # position
                        *ndir,  # direction
                        tod,  # time of day
                        o_name == "fridge",  # open(fridge_door)
                        o_name == "kitchen_unit" and side == "left",  # open(cupboard_door_left)
                        o_name == "kitchen_unit" and side == "right",  # open(cupboard_door_right)
                        o_name == "kitchen_unit" and side == "drawer",  # open(kitchen_unit_drawer)
                        o_name == "fridge",  # stove_door_open
                        o_name == "kitchen_unit" and tod != "morning" and side == "left",  # see cup
                        o_name == "kitchen_unit" and side == "drawer",  # see cutlery
                        o_name == "kitchen_unit" and tod != "morning" and side == "left",  # see bowl
                        o_name == "kitchen_unit",  # see sink
                        o_name == "fridge" and tod != "morning",  # see milk
                        o_name == 'fridge',  # see beer
                        o_name == "kitchen_unit" and tod != "morning" and side == "right",  # see cereal
                        o_name == 'stove',  # see stovetop
                        o_name == 'stove',  # see pot
                        o_name,
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


def plot_data(fp, name, varx, vary) -> go.Figure:
    logger.debug('plotting data...')

    df = pd.read_parquet(os.path.join(fp, 'data', f'000-{name}.parquet'))

    fig_d = px.scatter(
        df,
        x=varx,
        y=vary,
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
