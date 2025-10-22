import os
from jpt.base.intervals import ContinuousSet
from bayrob.core.base import BayRoB, Search
from bayrob.utils import locs
preset = {
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

bayrob = BayRoB()
bayrob.adddatapath([os.path.join(locs.examples, 'demo', d) for d in os.listdir(os.path.join(locs.examples, 'demo'))])
allvars = bayrob.models['move'].variables + \
                  bayrob.models['turn'].variables + \
                  bayrob.models['perception'].variables
allvars_ = {v.name: v for v in allvars}

asr = Search()
asr.bwd = preset['bwd']
asr.init = {allvars_[k]: v for k, v in preset['init'].items()}
asr.init_tolerances = {allvars_[k]: v for k, v in preset['init_tolerances'].items()}
asr.goal = {allvars_[k]: v for k, v in preset['goal'].items()}
asr.goal_tolerances = {allvars_[k]: v for k, v in preset['goal_tolerances'].items()}

bayrob.query = asr
bayrob.search_astar()
seq = bayrob.result.result
print(seq)