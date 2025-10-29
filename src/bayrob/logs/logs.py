import datetime
import os

import dnutils

from bayrob import config
from bayrob.utils import locs
from bayrob.utils.constants import bayroblogger, bayrobfileloggerv, bayrobfilelooger, \
    FILESTRFMT, FILESTRFMT_NOTIME, resultlog, logs

LEVELS = {'debug': dnutils.DEBUG,
          'info': dnutils.INFO,
          'warning': dnutils.WARNING,
          'error': dnutils.ERROR,
          'critical': dnutils.CRITICAL,
          }


def init_loggers(level='info') -> None:
    try:
        if 'mongo' in config and config.getboolean('mongo', 'available', fallback=False):

            loggers = {
                bayroblogger: dnutils.newlogger(dnutils.logs.console,
                                                level=LEVELS.get(level)),
                bayrobfilelooger: dnutils.newlogger(dnutils.logs.FileHandler(os.path.join(locs.logs, resultlog.format(datetime.datetime.now().strftime(FILESTRFMT)))), level=LEVELS.get(level)),
                bayrobfileloggerv: dnutils.newlogger(dnutils.logs.FileHandler(os.path.join(locs.logs, resultlog.format(datetime.datetime.now().strftime(FILESTRFMT)))), level=LEVELS.get(level)),
                'datalogger': dnutils.newlogger(dnutils.logs.FileHandler(os.path.join(locs.logs, f'{datetime.datetime.now().strftime(FILESTRFMT)}.csv')), level=LEVELS.get(level))
            }
        else:
            loggers = {
                bayroblogger: dnutils.newlogger(dnutils.logs.console,
                                                dnutils.logs.FileHandler(os.path.join(locs.logs, logs.format(datetime.datetime.now().strftime(FILESTRFMT_NOTIME)))),
                                                level=LEVELS.get(level)),
            }
    except:
        print(f'Could not initialize loggers')
        loggers = {}
    dnutils.loggers(loggers)
