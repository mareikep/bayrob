import os

import appdirs


# bayrob-dev/src/bayrob/
from bayrob.utils.constants import APPNAME, APPAUTHOR

code_base = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

# ~/.local/share/bayrob/
user_data = appdirs.user_data_dir(APPNAME, APPAUTHOR)

if os.path.basename(code_base).startswith('bayrob'):
    # bayrob-dev/src/
    src = os.path.realpath(os.path.join(code_base, '..'))

    # bayrob-dev/
    app_data = os.path.realpath(os.path.join(src, '..'))
else:
    # bayrob-dev/src/
    src = code_base

    # /usr/share/ubuntu/bayrob
    app_data = appdirs.site_data_dir(APPNAME, APPAUTHOR)
    if not os.path.exists(app_data):
        # ~/.local/share/bayrob/
        app_data = user_data

trdparty = os.path.join(app_data, '3rdparty')
doc = os.path.join(app_data, 'doc')
logs = os.path.join(app_data, 'logs')
examples = os.path.join(app_data, 'examples')
web = os.path.join(code_base, 'web')
resource = os.path.join(web, 'resource')
