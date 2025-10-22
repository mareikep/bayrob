import os
from jpt.base.intervals import ContinuousSet
from bayrob.core.base import BayRoB, Query
from bayrob.utils import locs

preset = {
     "evidence": {
         'detected(milk)': False,
         'x_in': ContinuousSet(58, 68),
         'y_in': ContinuousSet(70, 80),
         'nearest_furniture': 'fridge'
     },
     "queryvars": ['daytime', 'open(fridge_door)']
 }

bayrob = BayRoB()
bayrob.adddatapath([os.path.join(locs.examples, 'demo', "perception")])
allvars_ = {v.name: v for v in bayrob.models['perception'].variables}

qo = Query()
qo.model = bayrob.models['perception']
qo.evidence = {allvars_[k]: v for k, v in preset['evidence'].items()}
qo.queryvars = [bayrob.models['perception'].varnames[k] for k in preset['queryvars']]

bayrob.query = qo
bayrob.query_jpts()
cond, post = bayrob.result.result
print(cond)
print(post)