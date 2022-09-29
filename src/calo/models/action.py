import math

import dnutils
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import randint, choice
from typing import List

from jpt.distributions import Gaussian

datalogger = dnutils.getlogger('datalogger')


class Move:
    # uncertainty for degrees and distance
    DEG_U = 5
    DIST_U = 0.05

    # desired distance moved in one step
    STEPSIZE = 1

    def __init__(self, degu=5, distu=0.05):
        Move.DEG_U = degu
        Move.DIST_U = distu

    @staticmethod
    def rotate(x, y, deg) -> (float, float):
        deg = np.radians(-deg)
        return x * math.cos(deg) - y * math.sin(deg), x * math.sin(deg) + y * math.cos(deg)

    @staticmethod
    def turnleft(agent) -> None:
        Move.turndeg(agent, -90)

    @staticmethod
    def turnright(agent) -> None:
        Move.turndeg(agent, 90)

    @staticmethod
    def turndeg(agent, deg=45) -> None:
        g = Gaussian(deg, abs(Move.DEG_U * deg / 180))
        agent.dir = Move.rotate(agent.dirx, agent.diry, g.sample(1))

    @staticmethod
    def moveforward(agent, dist=1) -> None:
        p_ = agent.pos
        for i in range(dist):
            Move.movestep(agent)

        datalogger.debug(p_, agent.pos, dist)

    @staticmethod
    def movestep(agent) -> None:
        g = Gaussian(Move.STEPSIZE, Move.DIST_U)
        dist = g.sample(1)
        agent.collided = agent.world.collides([agent.x + agent.dirx * dist, agent.y + agent.diry * dist])
        if not agent.collided:
            agent.pos = agent.x + agent.dirx * dist, agent.y + agent.diry * dist


    @staticmethod
    def sampletrajectory(agent, actions=None, p=None, steps=10) -> np.ndarray:
        if p is None:
            p = []
        if actions is None:
            actions = []

        poses = []
        for i in range(steps):
            action = choice(actions, replace=False, p=p)
            action(agent)
            plt.scatter(*agent.pos, marker='*', label=f'Pos {i} ({action.__name__})', c='k')
            poses.append(agent.pos + agent.dir)

        poses = np.array(poses)
        plt.plot(poses[:, 0], poses[:, 1], label='Trajectory')
        plt.grid()
        plt.legend()
        plt.show()

        return poses


class TrajectorySimulation:
    def __init__(self, x=10, y=10, probx=None, proby=None):
        self._deltax = 1
        self._deltay = 1
        self._sizex = x
        self._sizey = y
        self._dirx = self._diry = self.x = self.y = 0
        self._dirxmap = {-1: 'left', 0: 'no', 1: 'right'}
        self._dirymap = {-1: 'down', 0: 'no', 1: 'up'}
        self._probx = probx or [.4, .2, .4]
        self._proby = proby or [.4, .2, .4]

    def _initpos(self, x, y) -> List[float]:
        # select normally distributed initial position on field
        x_ = [i for i in range(x)]
        y_ = [i for i in range(y)]
        newx = int(min(self._sizex, max(0, np.random.normal(np.mean(x_), np.std(x_)))))
        newy = int(min(self._sizey, max(0, np.random.normal(np.mean(y_), np.std(y_)))))
        return [newx, newy, self._dirxmap.get(0), self._dirymap.get(0)]

    def dir(self, prob) -> int:
        # select horizontal or vertical direction according to given distribution
        return choice([-1, 0, 1], replace=False, p=prob)

    def step(self, posx, posy) -> List[float]:
        # determine new position by adding delta in direction according to distributions; limit by field boundaries
        newx = min(self._sizex, max(0, posx + self._deltax * self.dir(self._probx)))
        newy = min(self._sizey, max(0, posy + self._deltay * self.dir(self._proby)))
        return [newx, newy, self._dirxmap.get(newx-posx), self._dirymap.get(newy-posy)]

    def sample(self, n=1, s=10, initpos=None) -> pd.DataFrame:
        # return n trajectories with s steps starting from either random or given positions
        # a given position can be used to connect trajectories

        if initpos is None:
            initpos = [self._initpos(self._sizex, self._sizey)]*n
        elif isinstance(initpos, (list, tuple)) and not isinstance(initpos[0], (list, tuple)):
            initpos = [initpos]*n
        if len(initpos) != n:
            raise Exception('Number of requested samples and number of given initial positions does not match!')

        for ip in range(n):
            samples = [initpos[ip]]
            for i in range(s-1):
                samples.append(self.step(*samples[-1][:2]))
            yield pd.DataFrame(samples, columns=['x', 'y', 'dirx', 'diry'])
