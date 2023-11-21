import ast
import datetime
import os
import sys

import argparse

import dnutils
import pandas as pd

from calo.logs.logs import init_loggers
from calo.utils import locs
from calo.utils.constants import calologger, FILESTRFMT
from calo.utils.utils import recent_example
from jpt import JPT, infer_from_dataframe
from jpt.base.intervals import ContinuousSet

try:
    import perception
    import robotaction
    import gridagent
except ModuleNotFoundError:
    sys.path.append(os.path.join('..', 'examples'))


logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


def learn_jpt(
        fp,
        args
):
    name = args.example

    logger.debug(f'learning constrained {name} JPT...')

    # learn discriminative JPT from data generated by test_data_curation for PERCEPTION
    df = pd.read_parquet(
        os.path.join(fp, 'data', f'000-{name}.parquet'),
    )

    logger.debug('Got dataframe of shape:', df.shape)

    constraints = args.constraints if 'constraints' in args else None
    if constraints is not None:
        c = [(var, op, v) for var, val in constraints.items() for v, op in ([(val.lower, ">="), (val.upper, "<=")] if isinstance(val, ContinuousSet) else [(val, "==")])]

        s = ' & '.join([f'({var} {op} {num})' for var, op, num in c])
        logger.debug('Extracting dataset using query: ', s)
        df = df.query(s)
        logger.debug('Returned subset of shape:', df.shape)

    variables = args.variables if 'variables' in args else None
    if variables is not None:
        logger.debug(f'Restricting dataset to columns: {variables}')
        df = df[variables]

    variables = infer_from_dataframe(
        df,
        scale_numeric_types=False,
        # precision=.5
    )

    # targets can be specified either by
    #   a) index or
    #   b) comma-separated string of variable names
    #   c) comma-separated string of variable names of features
    tgtidx = args.tgtidx if 'tgtidx' in args else None
    targets = args.targets if 'targets' in args else None
    features = args.features if 'features' in args else None
    tgts = []
    if tgtidx is not None:
        tgts = variables[int(tgtidx):]
    if targets is not None:
        tgts = [v for v in variables if v.name in targets]
    if features is not None:
        tgts = [v for v in variables if v.name not in features]

    jpt_ = JPT(
        variables=variables,
        targets=tgts,
        min_impurity_improvement=args.min_impurity_improvement,
        min_samples_leaf=args.min_samples_leaf  # .005
    )

    jpt_.learn(df, close_convex_gaps=False)
    # jpt_mf = jpt_mf.prune(similarity_threshold=.77)
    # jpt_mf.postprocess_leaves()

    logger.debug(f'...done! saving to file {os.path.join(fp, f"000-{name}.tree")}')

    jpt_.save(os.path.join(fp, f'000-{name}.tree'))

    logger.debug('...done.')


def plot_jpt(fp, args):
    name = args.example

    logger.debug(f'plotting {name} tree without distributions...')
    jpt_ = JPT.load(os.path.join(fp, f'000-{name}.tree'))
    jpt_.plot(
        title=name,
        filename=f'000-{name}-nodist',
        directory=os.path.join(fp, 'plots'),
        leaffill='#CCDAFF',
        nodefill='#768ABE',
        alphabet=True,
        view=args.showplots
    )

    logger.debug(f'plotting {name} tree...')
    jpt_.plot(
        title=name,
        plotvars=list(jpt_.variables),
        filename=f'000-{name}',
        directory=os.path.join(fp, 'plots'),
        leaffill='#CCDAFF',
        nodefill='#768ABE',
        alphabet=True,
        view=args.showplots
    )


def main(DT, args):

    fp = os.path.join(locs.examples, args.example, DT)

    if not os.path.exists(fp):
        os.mkdir(fp)
        os.mkdir(os.path.join(fp, 'plots'))
        os.mkdir(os.path.join(fp, 'data'))

    if args.example == 'robotaction_turn':
        from robotaction_turn import robotaction_turn as mod
    elif args.example == 'robotaction_move':
        from robotaction_move import robotaction_move as mod
    elif args.example == 'perception':
        from perception import perception as mod
    else:
        from perception import perception as mod
        args.example = 'perception'

    mod.init(fp, args)

    if not args.recent:
        mod.generate_data(fp, args)

    if args.learn:
        learn_jpt(fp, args)

    if args.plot:
        plot_jpt(fp, args)

    if args.data:
        mod.plot_data(fp, args)

    mod.teardown(fp, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CALOWeb.')
    parser.add_argument("-v", "--verbose", dest="verbose", default='debug', type=str, action="store", help="Set verbosity level {debug,info,warning,error,critical}. Default is info.")
    parser.add_argument('-r', '--recent', default=False, action='store_true', help='use most recent folder greated', required=False)
    parser.add_argument('-l', '--learn', default=False, action='store_true', help='learn model', required=False)
    parser.add_argument('-p', '--plot', default=False, action='store_true', help='plot model', required=False)
    parser.add_argument('-s', '--showplots', default=False, action='store_true', help='show plots', required=False)
    parser.add_argument('-d', '--data', default=False, action='store_true', help='trigger generating data/world plots', required=False)
    parser.add_argument('-a', '--args', action='append', nargs=2, metavar=('arg', 'value'), help='other, example-specific argument of type (arg, value)')
    parser.add_argument('-e', '--example', type=str, default='perception', help='name of the data set', required=False)
    parser.add_argument('-m', '--min-samples-leaf', type=float, default=.01, help='min_samples_leaf parameter', required=False)
    parser.add_argument('-i', '--min-impurity_improvement', type=float, default=.01, help='impurity_improvement parameter', required=False)
    parser.add_argument('-o', '--obstacles', default=False, action='store_true', help='obstacles', required=False)
    args = parser.parse_args()

    init_loggers(level=args.verbose)

    if args.args is not None:
        d = vars(args)
        for k, v in args.args:
            try:
                d[k] = ast.literal_eval(v)
            except:
                d[k] = v

    logger.info(f"Running example {args.example} with arguments: \n{', '.join(f'{k}={v}' for k, v in d.items() if k != 'args')})")

    # use most recently created dataset or create from scratch
    if args.recent:
        DT = recent_example(os.path.join(locs.examples, 'perception'))
        logger.debug(f'Using recent directory {DT}')
    else:
        DT = f'{datetime.datetime.now().strftime(FILESTRFMT)}'
        logger.debug(f'Creating new directory {DT}')

    main(DT, args)