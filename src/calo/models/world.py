from random import uniform

import numpy as np
from typing import List, Any, Dict, Tuple


class World:
    def __init__(self) -> None:
        pass


class Agent:
    def __init__(
            self,
            world: World = None
    ) -> None:
        self._world = world


class Grid(World):
    def __init__(
            self,
            x=[-100, 100],
            y=[-100, 100]
    ) -> None:
        super().__init__()
        self.coords = [x[0], y[0], x[1], y[1]]
        self._obstacles = []
        self._obstaclenames = []

    def obstacle(
            self,
            x0,
            y0,
            x1,
            y1,
            name: str = None
    ) -> None:
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
        self._obstaclenames.append(name)

    @property
    def obstacles(self) -> List[List[float]]:
        return self._obstacles

    @property
    def obstaclenames(self) -> List[List[float]]:
        return self._obstaclenames

    def collides_obstacle(
            self,
            pos: Tuple[float, float]
    ) -> bool:
        return any([o[0] <= pos[0] <= o[2] and o[1] <= pos[1] <= o[3] for o in self._obstacles])

    def collides_wall(
            self,
            pos: Tuple[float, float]
    ) -> bool:
        return not (self.coords[0] <= pos[0] <= self.coords[2] and self.coords[1] <= pos[1] <= self.coords[3])

    def collides(
            self,
            pos: Tuple[float, float]
    ) -> bool:
        # check obstacle collision AND wall collision
        return self.collides_obstacle(pos) or self.collides_wall(pos)


class GridAgent(Agent):
    def __init__(
            self,
            world: Grid = None,
            pos: Tuple[float, float] = None,
            dir: Tuple[float, float] = None
    ) -> None:
        super().__init__(world=world)
        self._posx = self._posy = 0
        self._dirx = self._diry = 0
        self.pos = pos
        self.dir = dir
        self._collided = False

    def __str__(self) -> str:
        return f'{self.pos} -> {self.dir}'

    @property
    def pos(self) -> (float, float):
        return self._posx, self._posy

    @pos.setter
    def pos(
            self,
            pos: Tuple[float, float] = None
    ) -> None:
        if pos is None:
            self._posx, self._posy = (None, None)
        else:
            self._posx, self._posy = pos

    @property
    def dir(self) -> (float, float):
        return self._dirx, self._diry

    @dir.setter
    def dir(
            self,
            dir: Tuple[float, float] = None
    ) -> None:
        if dir is None:
            self._dirx, self._diry = (None, None)
        else:
            # always set normalized direction vector
            self._dirx, self._diry = dir/np.linalg.norm(dir)

    @property
    def x(self) -> float:
        return self._posx

    @property
    def y(self) -> float:
        return self._posy

    @property
    def dirx(self) -> float:
        return self._dirx

    @property
    def diry(self) -> float:
        return self._diry

    @property
    def world(self) -> Grid:
        return self._world

    @world.setter
    def world(
            self,
            w: Grid
    ) -> None:
        self._world = w

    @property
    def collided(self) -> bool:
        return self._collided

    @collided.setter
    def collided(
            self,
            c: bool
    ) -> None:
        self._collided = c

    def init_random(self):
        initdir = (
            uniform(-1, 1),
            uniform(-1, 1)
        )
        initpos = (
            uniform(self.world.coords[0], self.world.coords[2]),
            uniform(self.world.coords[1], self.world.coords[3])
        )
        if not self.world.collides(initpos):
            self.pos = initpos
            self.dir = initdir
        else:
            self.init_random()
