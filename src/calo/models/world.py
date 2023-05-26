import numpy as np
from typing import List, Any, Dict


class World:
    def __init__(self) -> None:
        pass


class Grid(World):
    def __init__(self, x=100, y=100) -> None:
        super().__init__()
        self.coords = 0, 0, x, y
        self._obstacles = []

    def obstacle(self, x0, y0, x1, y1) -> None:
        # make sure x0/y0 are bottom left corner, x1/y1 upper right for collision check
        if x1 < x0:
            tmp = x1
            x1 = x0
            x0 = tmp
        if y1 < y0:
            tmp = y1
            y1 = y0
            y0 = tmp
        self._obstacles.append([x0, y0, x1, y1])

    @property
    def obstacles(self) -> List[List[float]]:
        return self._obstacles

    def collides(self, pos) -> bool:
        return any([o[0] <= pos[0] <= o[2] and o[1] <= pos[1] <= o[3] for o in self._obstacles])


class Agent:
    def __init__(self, pos, dir) -> None:
        self._x = self._y = 0
        self._dirx = self._diry = 0
        self.pos = pos
        self.dir = dir
        self._world = None
        self._collided = False

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

    @property
    def world(self) -> World:
        return self._world

    @world.setter
    def world(self, w) -> None:
        self._world = w

    @property
    def collided(self) -> bool:
        return self._collided

    @collided.setter
    def collided(self, c) -> None:
        self._collided = c
