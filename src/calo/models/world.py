import numpy as np


class Agent:
    def __init__(self, pos, dir):
        self._x = self._y = 0
        self._dirx = self._diry = 0
        self.pos = pos
        self.dir = dir

    def __str__(self) -> str:
        return f'{self.pos} -> {self.dir}'

    @property
    def pos(self) -> (float, float):
        return self._x, self._y

    @pos.setter
    def pos(self, pos) -> None:
        self._x, self._y = pos

    @property
    def dir(self) -> (float, float):
        return self._dirx, self._diry

    @dir.setter
    def dir(self, dir) -> None:
        # always set normalized direction vector
        self._dirx, self._diry = dir/np.linalg.norm(dir)

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def dirx(self) -> float:
        return self._dirx

    @property
    def diry(self) -> float:
        return self._diry
