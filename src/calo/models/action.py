import math
from typing import List, Dict, Tuple, Any

import dnutils
import jpt
import numpy as np
import pandas as pd
from dnutils import ifnone
from jpt.distributions import Gaussian
from matplotlib import pyplot as plt
from numpy.random import randint, choice

datalogger = dnutils.getlogger('datalogger')


class Move:
    # uncertainty for degrees and distance
    DEG_U = .01
    DIST_U = .05

    # desired distance moved in one step
    STEPSIZE = 1

    def __init__(
            self,
            degu: float = .01,
            distu: float = .05
    ):
        Move.DEG_U = degu
        Move.DIST_U = distu

    @staticmethod
    def rotate(
            x: float,
            y: float,
            deg: float
    ) -> (float, float):
        deg = np.radians(-deg)
        newdir = x * math.cos(deg) - y * math.sin(deg), x * math.sin(deg) + y * math.cos(deg)
        return newdir

    @staticmethod
    def turnleft(
            agent
    ) -> None:
        Move.turndeg(agent, -90)

    @staticmethod
    def turnright(
            agent
    ) -> None:
        Move.turndeg(agent, 90)

    @staticmethod
    def turndeg(
            agent,
            deg=45
    ) -> None:
        g = Gaussian(deg, math.pow(Move.DEG_U * deg,2))
        agent.dir = Move.rotate(agent.dirx, agent.diry, g.sample(1))

    @staticmethod
    def moveforward(
            agent,
            dist=1
    ) -> None:
        p_ = agent.pos
        for i in range(dist):
            Move.movestep(agent)

        datalogger.debug(p_, agent.pos, dist)

    @staticmethod
    def movestep(
            agent
    ) -> None:
        g = Gaussian(Move.STEPSIZE, Move.DIST_U)
        dist = g.sample(1)
        agent.collided = agent.world.collides([agent.x + agent.dirx * dist, agent.y + agent.diry * dist])
        if not agent.collided:
            agent.pos = agent.x + agent.dirx * dist, agent.y + agent.diry * dist


    @staticmethod
    def sampletrajectory(
            agent,
            actions=None,
            p=None,
            steps=10
    ) -> np.ndarray:
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

    @staticmethod
    def plot(
            jpt_: jpt.trees.JPT,
            qvarx: jpt.variables.Variable,
            qvary: jpt.variables.Variable,
            evidence: Dict[jpt.variables.Variable, Any] = None,
            title: str = None,
            conf: float = None,
            limx: Tuple = None,
            limy: Tuple = None,
            limz: Tuple = None,
            save: str = None,
            show: bool = False
    ) -> None:
        """Plots a heatmap representing the overall `coverage` of the jpt for the given variables, i.e. the joint
        probability of these variables: P(qvarx, qvary [| evidence ]). Helps to identify areas not well represented
        by the tree.

        :param jpt_: The (conditional) tree to plot the overall coverage for
        :param qvarx: The first of two joint variables to show the coverage for
        :param qvary: The second of two joint variable to show the coverage for
        :param evidence: The evidence for the conditional probability represented (if present)
        :param title: The plot title
        :param conf:  A confidence value. Values below this threshold are set to 0. (= equal color for lowest value in plot)
        :param limx: The limits for the x-variable; determined from pdf intervals of jpt priors if not given
        :param limy: The limits for the y-variable; determined from pdf intervals of jpt priors if not given
        :param limz: The limits for the z-variable; determined from pdf intervals of jpt priors if not given
        :param save: The location where the plot is saved (if given)
        :param show: Whether the plot is shown
        :return: None
        """
        from jpt.base.utils import format_path

        # determine limits
        xmin = ifnone(limx, jpt_.priors[qvarx.name].pdf.intervals[0].upper, lambda x: x[0])  # get limits for qvarx variable
        xmax = ifnone(limx, jpt_.priors[qvarx.name].pdf.intervals[-1].lower, lambda x: x[1])  # get limits for qvarx variable
        ymin = ifnone(limy, jpt_.priors[qvary.name].pdf.intervals[0].upper, lambda x: x[0])  # get limits for qvary variable
        ymax = ifnone(limy, jpt_.priors[qvary.name].pdf.intervals[-1].lower, lambda x: x[1])  # get limits for qvary variable

        # generate datapoints
        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)

        X, Y = np.meshgrid(x, y)
        Z = np.array(
            [
                jpt_.pdf(
                    jpt_.bind({
                        qvarx: x,
                        qvary: y
                    })
                ) for x, y, in zip(X.ravel(), Y.ravel())
            ]).reshape(X.shape)

        zmin = ifnone(limz, Z.min(), lambda x: x[0])
        zmax = ifnone(limz, Z.max(), lambda x: x[1])

        # show only values above a certain threshold, consider lower values as high-uncertainty areas
        if conf is not None:
            Z[Z < conf] = 0.

        # init plot
        fig, ax = plt.subplots(num=1, clear=True)
        fig.patch.set_facecolor('#D6E7F8')  # set bg color around the plot area (royal purple)
        ax.set_facecolor('#B1FF49')  # set bg color of plot area (dark purple)
        cmap = 'BuPu'  # viridis, Blues, PuBu, 0rRd, BuPu

        # generate heatmap
        c = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=zmin, vmax=zmax)
        ax.set_title(f'P({qvarx.name}, {qvary.name}{f"|{format_path(evidence, precision=3)}" if evidence else ""})')

        # setting the limits of the plot to the limits of the data
        ax.axis([xmin, xmax, ymin, ymax])
        ax.set_xlabel(f'{qvarx.name}')
        ax.set_ylabel(f'{qvary.name}')
        fig.colorbar(c, ax=ax)
        fig.suptitle(title)
        fig.canvas.manager.set_window_title('A* JPT')

        if save:
            plt.savefig(save)

        if show:
            plt.show()


class TrajectorySimulation:
    def __init__(
            self,
            x=10,
            y=10,
            probx=None,
            proby=None
    ):
        self._deltax = 1
        self._deltay = 1
        self._sizex = x
        self._sizey = y
        self._dirx = self._diry = self.x = self.y = 0
        self._dirxmap = {-1: 'left', 0: 'no', 1: 'right'}
        self._dirymap = {-1: 'down', 0: 'no', 1: 'up'}
        self._probx = probx or [.4, .2, .4]
        self._proby = proby or [.4, .2, .4]

    def _initpos(
            self,
            x,
            y
    ) -> List[float]:
        # select normally distributed initial position on field
        x_ = [i for i in range(x)]
        y_ = [i for i in range(y)]
        newx = int(min(self._sizex, max(0, np.random.normal(np.mean(x_), np.std(x_)))))
        newy = int(min(self._sizey, max(0, np.random.normal(np.mean(y_), np.std(y_)))))
        return [newx, newy, self._dirxmap.get(0), self._dirymap.get(0)]

    def dir(
            self,
            prob
    ) -> int:
        # select horizontal or vertical direction according to given distribution
        return choice([-1, 0, 1], replace=False, p=prob)

    def step(
            self,
            posx,
            posy
    ) -> List[float]:
        # determine new position by adding delta in direction according to distributions; limit by field boundaries
        newx = min(self._sizex, max(0, posx + self._deltax * self.dir(self._probx)))
        newy = min(self._sizey, max(0, posy + self._deltay * self.dir(self._proby)))
        return [newx, newy, self._dirxmap.get(newx-posx), self._dirymap.get(newy-posy)]

    def sample(
            self,
            n=1,
            s=10,
            initpos=None
    ) -> pd.DataFrame:
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
