import datetime
import os
from random import randint

import dnutils
import pandas as pd
from matplotlib import pyplot as plt

from calo.logs.logs import init_loggers
from calo.models.action import Move
from calo.models.world import Agent, Grid
from calo.utils import locs
from calo.utils.constants import FILESTRFMT, calologger
from jpt import infer_from_dataframe, JPT

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)
RUNS = 1000
ACTIONS = 100


def robot_pos_random(dt):
    logger.debug('Generating random robot data...')

    # init agent and world
    w = Grid()
    w.obstacle(25, 25, 50, 50)
    w.obstacle(-10, 10, 0, 40)
    w.obstacle(50, -30, 20, 10)
    w.obstacle(-75, -10, -50, -40)
    w.obstacle(-25, -50, -15, -75)

    # write sample data for MOVEFORWARD and TURN action of robot (absolute positions)
    for j in range(RUNS):
        poses = []  # for plotting
        turns = []

        a = Agent([0, 0], [1, 0])
        a.world = w

        for action in range(ACTIONS):
            deg = randint(-180, 180)
            turns.append(a.dir + (deg,))
            Move.turndeg(a, deg)

            steps = randint(1, 10)
            # poses.append(a.pos + a.dir + (steps, a.collided))
            # Move.moveforward(a, steps)
            for _ in range(steps):
                poses.append(a.pos + a.dir + (1, a.collided))
                Move.moveforward(a, 1)

        poses.append(a.pos + a.dir + (0, a.collided))
        turns.append(a.dir + (0,))

        df_moveforward = pd.DataFrame(poses, columns=['x', 'y', 'xdir', 'ydir', 'numsteps', 'collided'])
        df_moveforward.to_csv(os.path.join(locs.examples, 'robotaction', dt, f'{j}-MOVEFORWARD.csv'), index=False)

        df_turn = pd.DataFrame(turns, columns=['xdir', 'ydir', 'angle'])
        df_turn.to_csv(os.path.join(locs.examples, 'robotaction', dt, f'{j}-TURN.csv'), index=False)

        plt.scatter(df_moveforward['x'], df_moveforward['y'], marker='*', c='cornflowerblue')
        plt.plot(df_moveforward['x'], df_moveforward['y'], c='cornflowerblue')
        plt.scatter(df_moveforward['x'].iloc[0], df_moveforward['y'].iloc[0], marker='o', c='green', label='Start')
        plt.scatter(df_moveforward['x'].iloc[-1], df_moveforward['y'].iloc[-1], marker='o', c='red', label='End', zorder=1000)
        plt.savefig(os.path.join(locs.examples, 'robotaction', dt, f'{j}-MOVE.png'), format="png")  # TODO: remove to save storage space and prevent overload of produced images

    for i, o in enumerate(w.obstacles):
        plt.annotate(f'O_{i}', (o[0] + (o[2] - o[0]) / 2, o[1] + (o[3] - o[1]) / 2))
    plt.savefig(os.path.join(locs.examples, 'robotaction', dt, f'ALL-MOVE.png'), format="png")

    logger.debug('...done! Generating plot...')

    plt.grid()
    plt.legend()
    plt.show()


def data_curation(dt):
    logger.debug('curating robot MOVEFORWARD data...')

    # read position data files generated by test_robot_pos and generate large file containing deltas (position-independent)
    # FIXME: use collided once JSON error (bool not JSON serializable) is fixed
    # data_moveforward = pd.DataFrame(columns=['x_in', 'y_in', 'x_out', 'y_out', 'collided', 'numsteps'])
    # data_moveforward = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'x_in', 'y_in', 'x_out', 'y_out', 'numsteps'])
    data_moveforward = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'x_in', 'y_in', 'x_out', 'y_out'])
    for i in range(RUNS):
        with open(os.path.join(locs.examples, 'robotaction', dt, f'{i}-MOVEFORWARD.csv'), 'r') as f:
            d = pd.read_csv(f, delimiter=',', header=0)
            for idx, row in d.iterrows():
                if idx == d.index.max(): break
                # FIXME: use collided once JSON error (bool not JSON serializable) is fixed
    #             # data_moveforward.loc[idx + i * 100] = [row['x'], row['y'], d.iloc[idx + 1]['x'], d.iloc[idx + 1]['y'], row['collided'], row['numsteps']]
    #             data_moveforward.loc[idx + i * RUNS] = [row['xdir'], row['ydir'], row['x'], row['y'], d.iloc[idx + 1]['x'], d.iloc[idx + 1]['y'], row['numsteps']]
                data_moveforward.loc[idx + i * RUNS] = [row['xdir'], row['ydir'], row['x'], row['y'], d.iloc[idx + 1]['x'], d.iloc[idx + 1]['y']]
    data_moveforward.to_csv(os.path.join(locs.examples, 'robotaction', dt, f'ALL-MOVEFORWARD.csv'), index=False)

    logger.debug('...done! curating robot TURN data...')
    data_turn = pd.DataFrame(columns=['xdir_in', 'ydir_in', 'xdir_out', 'ydir_out', 'angle'])
    for i in range(RUNS):
        with open(os.path.join(locs.examples, 'robotaction', dt, f'{i}-TURN.csv'), 'r') as f:
            d = pd.read_csv(f, delimiter=',', header=0)
            for idx, row in d.iterrows():
                if idx == d.index.max(): continue
                # store data of two consecuting steps
                data_turn.loc[idx + i * RUNS] = [row['xdir'], row['ydir'], d.iloc[idx + 1]['xdir'], d.iloc[idx + 1]['ydir'], row['angle']]
    data_turn.to_csv(os.path.join(locs.examples, 'robotaction', dt, f'ALL-TURN.csv'), index=False)
    logger.debug('...done!')


def jpt_moveforward(dt):
    logger.debug('learning MOVEFORWARD JPT...')

    # learn discriminative JPT from data generated by test_data_curation for MOVEFORWARD
    data_moveforward = pd.read_csv(os.path.join(locs.examples, 'robotaction', dt, f'ALL-MOVEFORWARD.csv'), delimiter=',', header=0)
    movevars = infer_from_dataframe(data_moveforward, unique_domain_names=True, scale_numeric_types=False)
    # FIXME: use targets=movevars[2:5] once JSON error (bool not JSON serializable) is fixed
    jpt_mf = JPT(variables=movevars, targets=movevars[4:6], min_samples_leaf=int(data_moveforward.shape[0] * 0.008))
    # jpt_mf = JPT.load(os.path.join(locs.examples, 'robotaction', dt, f'MOVEFORWARD.tree'))
    jpt_mf.learn(columns=data_moveforward.values.T, keep_samples=True)
    jpt_mf.save(os.path.join(locs.examples, 'robotaction', dt, f'MOVEFORWARD.tree'))
    logger.debug('...done! Plotting MOVEFORWARD tree...')
    jpt_mf.plot(title='MOVEFORWARD', plotvars=jpt_mf.variables, filename=f'MOVEFORWARD', directory=os.path.join(locs.examples, 'robotaction', dt))
    logger.debug('...done!')


def jpt_turn(dt):
    logger.debug('learning TURN JPT...')

    # learn discriminative JPT from data generated by test_data_curation for TURN
    data_turn = pd.read_csv(os.path.join(locs.examples, 'robotaction', dt, f'ALL-TURN.csv'), delimiter=',', header=0)
    turnvars = infer_from_dataframe(data_turn, unique_domain_names=True, scale_numeric_types=False)
    jpt_t = JPT(variables=turnvars, targets=turnvars[2:4], min_samples_leaf=int(data_turn.shape[0] * 0.008))
    # jpt_t = JPT.load(os.path.join(locs.examples, 'robotaction', dt, f'TURN.tree'))
    jpt_t.learn(columns=data_turn.values.T, keep_samples=True)
    jpt_t.save(os.path.join(locs.examples, 'robotaction', dt, f'TURN.tree'))
    logger.debug('...done! Plotting TURN tree...')
    jpt_t.plot(title='TURN', plotvars=jpt_t.variables, filename=f'TURN', directory=os.path.join(locs.examples, 'robotaction', dt))
    logger.debug('...done!')

    
if __name__ == '__main__':
    init_loggers(level='debug')
    DT = f'{datetime.datetime.now().strftime(FILESTRFMT)}'
    # DT = f'2022-09-19_08:48'
    DT = f'2023-05-16_14:02'
    if not os.path.exists(os.path.join(locs.examples, 'robotaction', DT)):
        os.mkdir(os.path.join(locs.examples, 'robotaction', DT))
    logger.debug(f'Running robotaction data generation with timestamp {DT}')

    # robot_pos_random(DT)
    # data_curation(DT)
    jpt_moveforward(DT)
    jpt_turn(DT)
    