import datetime
import os

import dnutils
from src.calo import config

from calo.database import connection
from calo.utils import locs
from calo.utils.constants import calologger, calojsonlogger, calofileloggerv, calofileloggerr, \
    FILESTRFMT, resultlog, logs

LEVELS = {'debug': dnutils.DEBUG,
          'info': dnutils.INFO,
          'warning': dnutils.WARNING,
          'error': dnutils.ERROR,
          'critical': dnutils.CRITICAL,
          }


def init_loggers(level='info') -> None:
    if 'mongo' in config and config.getboolean('mongo', 'available', fallback=False):
        loggers = {
            calologger: dnutils.newlogger(dnutils.logs.console,
                                          dnutils.logs.MongoHandler(connection.logs.calologdebug, checkkeys=False), level=LEVELS.get(level)),
            calojsonlogger: dnutils.newlogger(dnutils.logs.MongoHandler(connection.logs.calolog, checkkeys=False), level=LEVELS.get(level)),
            calofileloggerr: dnutils.newlogger(dnutils.logs.FileHandler(os.path.join(locs.logs, resultlog.format(datetime.datetime.now().strftime(FILESTRFMT)))), level=LEVELS.get(level)),
            calofileloggerv: dnutils.newlogger(dnutils.logs.FileHandler(os.path.join(locs.logs, resultlog.format(datetime.datetime.now().strftime(FILESTRFMT)))), level=LEVELS.get(level)),
            'datalogger': dnutils.newlogger(dnutils.logs.FileHandler(os.path.join(locs.logs, f'{datetime.datetime.now().strftime(FILESTRFMT)}-MOVE.csv')), level=LEVELS.get(level))
        }
    else:
        loggers = {
            calologger: dnutils.newlogger(dnutils.logs.console, level=LEVELS.get(level)),
            calojsonlogger: dnutils.newlogger(dnutils.logs.console, level=LEVELS.get(level)),
            calofileloggerr: dnutils.newlogger(dnutils.logs.FileHandler(os.path.join(locs.logs, logs.format(datetime.datetime.now().strftime(FILESTRFMT)))), level=LEVELS.get(level)),
            calofileloggerv: dnutils.newlogger(dnutils.logs.FileHandler(os.path.join(locs.logs, logs.format(datetime.datetime.now().strftime(FILESTRFMT)))), level=LEVELS.get(level)),
            'datalogger': dnutils.newlogger(dnutils.logs.FileHandler(os.path.join(locs.logs, f'{datetime.datetime.now().strftime(FILESTRFMT)}-MOVE.csv')), level=LEVELS.get(level))
        }
    dnutils.loggers(loggers)
