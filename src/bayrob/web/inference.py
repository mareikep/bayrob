import argparse
import sys
import traceback

import dnutils

from bayrob.core.base import BayRoB
from bayrob.logs.logs import init_loggers

from bayrob.utils.constants import bayroblogger
from pyrap.threads import DetachedSessionThread
from bayrob.utils.errors import UnsupportedFormat

logger = dnutils.getlogger(bayroblogger)


class Inference(DetachedSessionThread):

    def __init__(self):
        DetachedSessionThread.__init__(self)
        self.calo = BayRoB()
        self._data = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d

    def infer(self, query, threshold=1e-10, strategy=1):
        self.calo.query = query
        self.calo.threshold = threshold
        self.calo.strategy = strategy
        self.calo.infer()

        candidates = self.calo.hypotheses
        logger.info('Inference result', candidates)

    def learn(self, name, model='tree', path='.'):
        '''
        :param name: the name of the model to be learned
        :param model:
        :param path:
        :type name: str
        :return:
        '''

        if self.data is None:
            raise ValueError('Data has not been initialized.')
        else:
            self.calo.learn(name, self.data, model=model, path=path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", dest="verbose", default=1, type=int, action="store", help="Set verbosity level {1..5}. Default is 1.")
    parser.add_argument("-n", "--name", dest="modelname", default='DefaultModel', type=str, action="store", help="The name of the model to be learned.")
    parser.add_argument("-d", "--dir", dest="subdir", default='.', type=str, action="store", help="The subdirectory of locs.models to store the model to.")
    parser.add_argument("-m", "--model", dest="model", default='tree', type=str, action="store", help="The type of the model to be learned. Valid inputs are: tree, mln. Comma-separate values if multiple models shall be learned.")
    args = parser.parse_args()

    init_loggers(level=args.verbose * 10)
    mi = Inference()
    data = []  # TODO: load default data
    try:
        logger.debug('Learning', args.modelname)
        mi.data = data
        mi.learn(args.modelname, model=args.model, path=args.subdir)
    except NotImplementedError:
        logger.warning('Learning for this Model not yet implemented. Skipping', args.modelname)
    except UnsupportedFormat:
        logger.error('Got unsupported format!', traceback.format_exc())
        traceback.print_exc()
    except KeyboardInterrupt:
        logger.warning('Quitting')
        sys.exit(1)
    except:
        logger.critical('Could not retrieve data for Model. Skipping', args.modelname, traceback.format_exc())
        traceback.print_exc()
