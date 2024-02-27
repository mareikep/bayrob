import traceback

import dnutils
import sys
from pymongo import MongoClient

from bayrob import config
from bayrob.utils.constants import connectionlogger
from bayrob.utils.errors import MongoError

logger = dnutils.getlogger(connectionlogger)


class DBConnection:
    """The Database Connection singleton.

    If the mongo db connection settings are defined in the default config file, these settings are used to set up
    the db connection. If not, `BayRoB` connects to the local default database.

    .. seealso:: :class:`bayrob.config.Config`

    """
    def __init__(self) -> None:
        self.mongodb = None
        try:
            # use user-defined mongo configuration, if available. Otherwise, use default configuration and create default db
            if 'mongo' in config:
                logger.info('Loading user-defined mongo configuration...')
                self.mongodb = MongoClient(
                    host=config.get('mongo', 'host', fallback='127.0.0.1'),
                    port=config.getint('mongo', 'port', fallback=27017),
                    username=config.get('mongo', 'user', fallback=None),
                    password=config.get('mongo', 'password', fallback=None),
                    authMechanism=config.get('mongo', 'auth', fallback='DEFAULT'),
                    serverSelectionTimeoutMS=config.get('mongo', 'timeout', fallback=3000))
            else:
                logger.info('Loading default mongo configuration...')
                self.mongodb = MongoClient(serverSelectionTimeoutMS=config.get('mongo', 'timeout', fallback=3000))
                db_ = self.mongodb["calodb"]
            try:
                # check if connection exists, if not, create new database
                self.mongodb.server_info()
                logger.info('Successfully connected to', self.mongodb)
                if not config.has_section('bayrob'):
                    config.add_section('bayrob')
                config.set('bayrob', 'available', 'true')
            except Exception as e:
                logger.error(f'Failed to connect to mongodb due to "{type(e).__name__}" error.')
                if not config.has_section('bayrob'):
                    config.add_section('bayrob')
                config.set('bayrob', 'available', 'false')
        except MongoError:
            logger.error('Failed to connect to mongodb.')
            if not config.has_section('bayrob'):
                config.add_section('bayrob')
            config.set('bayrob', 'available', 'false')
            traceback.print_exc()


db = DBConnection().mongodb

sys.modules[__name__] = db
