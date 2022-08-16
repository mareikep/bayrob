import sys

from configparser import ConfigParser

import os

import dnutils
from dnutils import ifnone

from calo.utils import locs
from calo.utils.constants import calologger, calodocUP, calodoclow

logger = dnutils.getlogger(calologger)


class Config(ConfigParser):
    f"""Global configuration data structure singleton for {calodocUP.__doc__}. 
    Wraps around a :class:`configparser.ConfigParser`.
    The config file may look like this::

        [mongo]
            host = dbserver.example.com
            auth = SCRAM-SHA-1
            password = password123
            user = username
            port = 27017

        [onto]
            path = /path/to/owl/file

        [upload]
            allowedext = .csv,.txt
            maxfilesize = 1e+7
            maxdirsize = 4e+6
            maxfilecnt = 10
            uploadfldr = /tmp

        [{calodoclow.__doc__}]
            smoothed = True
            public = False
    """
    DEFAULTS = {}

    def __init__(self, filename=None):
        """The default configuration file is called ``caloconf`` and is located in the src directory of the system.

        :param filename: the name of the configuration file.
        :type filename: str
        """
        ConfigParser.__init__(self, allow_no_value=True)
        for section, values in self.DEFAULTS.items():
            self.add_section(section)
            for key, value in values.items():
                self.set(section, key, value)
        if filename is not None:
            self.filename = filename
            self.read(filename)

    def write(self, filename=None, **kwargs) -> None:
        """
        Saves this configuration file to disk.

        :param filename:    the name of the config file.
        :param kwargs:      additional arguments
        """
        filename = ifnone(filename, 'caloconf')
        filepath = os.path.join(locs.app_data, filename)
        with open(filepath, 'w+') as f:
            ConfigParser.write(self, f)

    def getlist(self, section, key, separator='\n'):
        return filter(bool, [s.strip() for s in self.get(section, key).split(separator)])


config = Config(os.path.join(locs.user_data, 'caloconf'))
logger.info(f'Loading config file "{config.filename}"')

sys.modules[__name__] = config
