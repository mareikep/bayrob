############
# PRINTING #
############
cs = ',\n'
nl = '\n'
cst = '\n\t'

###########
# PROJECT #
###########
def projectnameUP(): 'BayRoB'
def projectnameLOW(): 'bayrob'
APPNAME = projectnameUP.__doc__
APPAUTHOR = 'picklum'

###########
# LOGGING #
###########
bayroblogger = f'/bayrob'
bayrobjsonlogger = f'/bayrob/json'
bayrobfilelooger = f'/bayrob/file/res'
bayrobfileloggerv = f'/bayrob/file/verbose'
connectionlogger = f'/bayrob/connection'

resultlog = f'{projectnameUP.__doc__}_res-{{}}.log'
logs = f'{projectnameUP.__doc__}_log-{{}}.log'

#################
# TIME AND DATE #
#################
TMPFILESTRFMT = 'TMP_%Y%m%d_%H-%M-%S'
FILESTRFMT = "%Y-%m-%d_%H:%M"
FILESTRFMT_NOTIME = "%Y-%m-%d"
FILESTRFMT_SEC = "%Y-%m-%d_%H:%M:%S"

#################
# TIME AND DATE #
#################

obstacle_kitchen_boundaries = ((0, 0, 100, 100), "kitchen_boundaries")
obstacles = [
    ((15, 10, 25, 20), "chair1"),
    ((35, 10, 45, 20), "chair2"),
    ((10, 30, 50, 50), "kitchen_island"),
    ((80, 30, 100, 70), "stove"),
    ((10, 80, 50, 100), "kitchen_unit"),
    ((60, 80, 80, 100), "fridge"),
]

searchpresets = {
    "short": {
        "init": {
            'x_in': 3.5,
            'y_in': 58.5,
            'xdir_in': .75,
            'ydir_in': .75,
        },
        "init_tolerances": {
            'x_in': .05,
            'y_in': .05,
            'xdir_in': .01,
            'ydir_in': .01,
        },
        "goal": {
            'x_in': 5,
            'y_in': 60
        },
        "goal_tolerances": {
            'x_in': .5,
            'y_in': .5
        },
        "bwd": False
    },
    "multinomial": {
        "init": {
            'x_in': 62,
            'y_in': 74,
            'xdir_in': .3,
            'ydir_in': .9,
        },
        "init_tolerances": {
            'x_in': .1,
            'y_in': .1,
            'xdir_in': .01,
            'ydir_in': .01,
        },
        "goal": {
            'detected(milk)': {True},
        },
        "goal_tolerances": {},
        "bwd": True
    }
}

querypresets = {
    'perception': {
        "first": {
            "evidence": {
                'detected(milk)': False,
                'x_in': '[58, 68]',
                'y_in': '[70, 80]',
                'nearest_furniture': 'fridge'
            },
            "queryvars": ['daytime', 'open(fridge_door)']
        },
        "second": {
            "evidence": {
                'detected(milk)': False,
                'x_in': '[58, 68]',
                'y_in': '[70, 80]',
                'nearest_furniture': 'fridge',
                'daytime': 'post-breakfast'
            },
            "queryvars": ['open(fridge_door)']
        },
        "third": {
            "evidence": {
                'x_in': '[10, 50]',
                'y_in': '[70, 80]'
            },
            "queryvars": [
                'detected(beer)',
                'detected(bowl)',
                'detected(cereal)',
                'detected(cup)',
                'detected(cutlery)',
                'detected(milk)',
                'detected(pot)',
                'detected(sink)',
                'detected(stovetop)'
            ]
        },
        "fourth": {
            "evidence": {
                'x_in': '[60, 80]',
                'y_in': '[70, 80]'
            },
            "queryvars": [
                'detected(beer)',
                'detected(bowl)',
                'detected(cereal)',
                'detected(cup)',
                'detected(cutlery)',
                'detected(milk)',
                'detected(pot)',
                'detected(sink)',
                'detected(stovetop)'
            ]
        }
    }
}
