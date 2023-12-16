import datetime
import glob
import os
from pathlib import Path

import dnutils
import numpy as np
import pandas as pd

from calo.utils import locs
from calo.utils.constants import calologger, FILESTRFMT
from calo.utils.utils import recent_example, angles_from_quaternion, euler_from_quaternion

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


def generate_data(fp, args):

    def angle(row):
        return {k: v for k, v in zip(['angle_x', 'angle_y', 'angle_z'], euler_from_quaternion(row['q_x'], row['q_y'], row['q_z'], row['q_w']))}

    # merge actions and poses
    for path in Path(os.path.join(locs.examples, 'pr2', 'raw')).glob('16*/'):
        df_a = pd.read_csv(os.path.join(path, "actions.csv"), delimiter=";")
        df_p = pd.read_csv(os.path.join(path, "poses.csv"), delimiter=";")
        df = pd.merge(df_a, df_p, left_on='id', right_on='action_task_id', how="outer")

        # drop empty columns next, previous, object_type, grasp and effort as well as uninformative cols id_y
        # and action_task_id
        df = df.drop(columns=[cn for cn, e in zip(df.columns, df.isnull().all().values) if e] + ['id_y', 'action_task_id'])

        # set default values for empty cells
        df.fillna({"failure": "None",
                         "object_acted_on": "None",
                         "bodyPartsUsed": "None",
                         "arm": "None",
                         "t_x": 0,
                         "t_y": 0,
                         "t_z": 0,
                         "q_x": 0,
                         "q_y": 0,
                         "q_z": 0,
                         "q_w": 1,
                         "information": "None"
                    }, inplace=True)

        # add angle columns (degrees) calculated from quaternion
        df = pd.concat([df, pd.DataFrame.from_records(df.apply(angle, axis=1))], axis=1)

        # drop columns that are not necessary anymore (quaternion)
        df = df.drop(columns=['q_x', 'q_y', 'q_z', 'q_w'])

        logger.debug(f"saving merged file to {os.path.join(fp, 'data', f'{path.name}.parquet')}")
        df.to_parquet(os.path.join(fp, 'data', f'{path.name}.parquet'))
        df.to_csv(os.path.join(fp, 'data', f'{path.name}.csv'))

    # generate cumulated data file
    all_files = glob.glob(os.path.join(fp, 'data', "*.parquet"))
    logger.debug(f"Merging {len(all_files)} files...")
    df = pd.concat((pd.read_parquet(f) for f in all_files), ignore_index=True)

    # set type for data columns
    cols = {
        'id_x': str,
        'type': str,
        'startTime': np.float32,  # 'datetime64[s]',
        'endTime': np.float32,  # 'datetime64[s]',
        'duration': np.float32,
        'success': bool,
        'failure': str,
        'parent': str,
        'object_acted_on': str,
        'bodyPartsUsed': str,
        'arm': str,
        't_x': np.float32,
        't_y': np.float32,
        't_z': np.float32,
        'information': str,
        'angle_x': np.float32,
        'angle_y': np.float32,
        'angle_z': np.float32
    }
    df = df.astype(cols)

    # save data
    logger.debug(f"...done! Saving {df.shape[0]} data points to {os.path.join(fp, 'data', f'000-{args.example}.parquet')}...")
    df.to_parquet(os.path.join(fp, 'data', f'000-{args.example}.parquet'), index=False)
    df.to_csv(os.path.join(fp, 'data', f'000-{args.example}.csv'), index=False)

    return df


def plot_data(fp, args):
    pass


def init(fp, args):
    pass


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