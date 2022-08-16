import os

from typing import List

from core.base import CALO
from jpt import JPT
from pyrap.engine import PushService

from pyrap.threads import DetachedSessionThread


class CALODST(DetachedSessionThread):
    def __init__(self, webapp):
        DetachedSessionThread.__init__(self)
        self.pushsession = PushService()
        self.webapp = webapp
        self.callback = None
        self._calo = CALO()
        self.runfunction = 'inf_calo'
        self._f = {'infer': self.inf_calo}

    @property
    def query(self) -> dict:
        return self._calo.query

    @query.setter
    def query(self, query) -> None:
        self._calo.query = query

    @property
    def threshold(self) -> float:
        return self._calo.threshold

    @threshold.setter
    def threshold(self, threshold) -> None:
        self._calo.threshold = threshold

    @property
    def models(self) -> dict:
        return self._calo.models

    @models.setter
    def models(self, trees) -> None:
        self._calo.models = trees

    def adddatapath(self, path) -> None:
        self._calo.adddatapath(path)
        self._calo.reloadmodels()

    @property
    def strategy(self) -> int:
        return self._calo.strategy

    @strategy.setter
    def strategy(self, strategy) -> None:
        self._calo.strategy = strategy

    @property
    def resulttree(self) -> core.base.ResTree:
        return self._calo.resulttree

    @property
    def hypotheses(self) -> List[core.base.Hypothesis]:
        return self._calo.hypotheses

    @property
    def nodes(self) -> List[core.base.ResTree.Node]:
        return self._calo.nodes

    def allmodels(self, subdir=None) -> List[JPT]:
        fpath = os.path.join(self._datapath, subdir) if subdir is not None else self._datapath
        return [JPT.load(os.path.join(fpath, t)) for t in os.listdir(fpath) if t.endswith('.tree')]

    def inf_calo(self) -> None:
        self._calo.infer()

    def spoint(self) -> None:
        pass

    def run(self) -> None:
        self._f.get(self.runfunction, self.inf_calo)()

        # notify webapp that inference is finished.
        if self.callback is not None:
            self.callback(self.webapp)

        self.pushsession.flush()
        self.pushsession.stop()
