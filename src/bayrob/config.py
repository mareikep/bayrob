import sys

from configparser import ConfigParser

import os

import dnutils
from dnutils import ifnone

from bayrob.utils import locs
from bayrob.utils.constants import bayroblogger, projectnameUP, projectnameLOW

logger = dnutils.getlogger(bayroblogger)


class Config(ConfigParser):
    """Global configuration data structure singleton for BayRoB.
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

        [bayrob]
            smoothed = True
            public = False
    """
    DEFAULTS = {}

    def __init__(
            self,
            filename: str = None
    ):
        """The default configuration file is called ``caloconf`` and is located in the src directory of the system.

        :param filename: the name of the configuration file.
        """
        ConfigParser.__init__(self, allow_no_value=True)
        for section, values in self.DEFAULTS.items():
            self.add_section(section)
            for key, value in values.items():
                self.set(section, key, value)
        if filename is not None:
            self.filename = filename
            self.read(filename)

    def write(
            self,
            filename: str = None,
            **kwargs
    ) -> None:
        """
        Saves this configuration file to disk.

        :param filename:    the name of the config file.
        :param kwargs:      additional arguments
        """
        filename = ifnone(filename, 'caloconf')
        filepath = os.path.join(locs.app_data, filename)
        with open(filepath, 'w+') as f:
            ConfigParser.write(self, f)

    def getlist(
            self,
            section: str,
            key: str,
            separator: str = '\n'
    ):
        return filter(bool, [s.strip() for s in self.get(section, key).split(separator)])


config = Config(os.path.join(locs.user_data, 'caloconf'))
logger.info(f'Loading config file "{config.filename}"')

sys.modules[__name__] = config
