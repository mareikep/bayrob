import os
from pathlib import Path
from typing import Dict

import bayrob
from bayrob.core.base import BayRoB, Query
from jpt import JPT
from pyrap.engine import PushService
from pyrap.threads import DetachedSessionThread


class BayRoBSessionThread(DetachedSessionThread):
    def __init__(self, webapp):
        DetachedSessionThread.__init__(self)
        self.pushsession = PushService()
        self.webapp = webapp
        self.callback = None
        self._bayrob = BayRoB()
        self.runfunction = 'queryjpt'
        self._f = {
            'astar': self.astar,
            'queryjpt': self.query_jpts
        }

    @property
    def query(self) -> Query:
        return self._bayrob.query

    @query.setter
    def query(self, query) -> None:
        self._bayrob.query = query

    @property
    def models(self) -> dict:
        return self._bayrob.models

    @property
    def datasets(self) -> dict:
        return self._bayrob.datasets

    @models.setter
    def models(self, trees) -> None:
        self._bayrob.models = trees

    def adddatapath(self, path) -> None:
        self._bayrob.adddatapath(path)

    @property
    def result(self) -> bayrob.core.base.Result:
        return self._bayrob.result

    def allmodels(self, subdir=None) -> Dict[str, JPT]:
        models = {}
        for fpath in self._bayrob.datapaths:
            for treefile in Path(fpath).glob('*.tree'):
                models[treefile.name] = JPT.load(os.path.join(os.path.join(fpath, subdir) if subdir is not None else fpath, treefile))
        return models

    def query_jpts(self) -> None:
        self._bayrob.query_jpts()

    def astar(self) -> None:
        self._bayrob.search_astar()

    def run(self) -> None:
        self._f.get(self.runfunction, self.query_jpts)()

        # notify webapp that inference is finished.
        if self.callback is not None:
            self.callback(self.webapp)

        self.pushsession.flush()
        self.pushsession.stop()
