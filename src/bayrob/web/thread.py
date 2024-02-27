import os

from typing import List, Any

from bayrob.core.base import BayRoB
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
        self.runfunction = 'inf_bayrob'
        self._f = {'infer': self.inf_bayrob}

    @property
    def query(self) -> dict:
        return self._bayrob.query

    @query.setter
    def query(self, query) -> None:
        self._bayrob.query = query

    @property
    def threshold(self) -> float:
        return self._bayrob.threshold

    @threshold.setter
    def threshold(self, threshold) -> None:
        self._bayrob.threshold = threshold

    @property
    def models(self) -> dict:
        return self._bayrob.models

    @models.setter
    def models(self, trees) -> None:
        self._bayrob.models = trees

    def adddatapath(self, path) -> None:
        self._bayrob.adddatapath(path)
        self._bayrob.reloadmodels()

    @property
    def strategy(self) -> int:
        return self._bayrob.strategy

    @strategy.setter
    def strategy(self, strategy) -> None:
        self._bayrob.strategy = strategy

    @property
    def resulttree(self) -> bayrob.core.base.ResTree:
        return self._bayrob.resulttree

    @property
    def hypotheses(self) -> List[Any]:
        return self._bayrob.hypotheses

    @property
    def nodes(self) -> List[bayrob.core.base.ResTree.Node]:
        return self._bayrob.nodes

    def allmodels(self, subdir=None) -> List[JPT]:
        fpath = os.path.join(self._datapath, subdir) if subdir is not None else self._datapath
        return [JPT.load(os.path.join(fpath, t)) for t in os.listdir(fpath) if t.endswith('.tree')]

    def inf_bayrob(self) -> None:
        self._bayrob.infer()

    def spoint(self) -> None:
        pass

    def run(self) -> None:
        self._f.get(self.runfunction, self.inf_bayrob)()

        # notify webapp that inference is finished.
        if self.callback is not None:
            self.callback(self.webapp)

        self.pushsession.flush()
        self.pushsession.stop()
