import datetime
import os
import pprint
import unittest
from random import randint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from calo.core.base import CALO
from calo.models.action import TrajectorySimulation, Move
from calo.models.world import Agent, Grid
from calo.utils import locs
from calo.utils.constants import FILESTRFMT
from jpt import infer_from_dataframe, JPT
from jpt.distributions import Gaussian


class JointProbabilityTreesMPE(unittest.TestCase):

    def setUpClass(cls) -> None:
        cls.dt = None

    def test_robot_single_action(self) -> None:
        a1 = TrajectorySimulation(1000, 1000, probx=[.39, .01, .6], proby=[.59, .01, .4])
        d1 = list(a1.sample(1, 500))[0]
        v1 = infer_from_dataframe(d1, scale_numeric_types=True)
        t1 = JPT(variables=v1, targets=v1[2:], min_samples_leaf=int(d1.shape[0] * 0.1 / len(v1)))
        t1.learn(columns=d1.values.T)

        plt.plot(d1['x'], d1['y'], label='Trajectory')
        plt.scatter(d1['x'].iloc[0], d1['y'].iloc[0], label='Start')
        plt.scatter(d1['x'].iloc[-1], d1['y'].iloc[-1], label='End')

        # plt.ylim(0, 100)
        # plt.xlim(0, 100)
        plt.grid()
        plt.legend()
        plt.show()

        # t1.plot(title=f'Single Action', plotvars=v1)

    def test_robot_multiple_actions_single_sample(self) -> None:
        a1 = TrajectorySimulation(1000, 1000, probx=[.39, .01, .6], proby=[.59, .01, .4])
        d1 = list(a1.sample(1, 50, initpos=[500, 500, 'no', 'no']))[0]
        v1 = infer_from_dataframe(d1, scale_numeric_types=True)
        t1 = JPT(variables=v1, targets=v1[2:], min_samples_leaf=int(d1.shape[0] * 0.1 / len(v1)))
        t1.learn(columns=d1.values.T)

        a2 = TrajectorySimulation(1000, 1000, probx=[.6, .1, .3], proby=[.2, .1, .7])
        d2 = list(a2.sample(1, 50, initpos=[d1['x'].iloc[-1], d1['y'].iloc[-1], d1['dirx'].iloc[-1], d1['diry'].iloc[-1]]))[0]
        v2 = infer_from_dataframe(d2, scale_numeric_types=True)
        t2 = JPT(variables=v2, targets=v2[2:], min_samples_leaf=int(d2.shape[0] * 0.1 / len(v2)))
        t2.learn(columns=d2.values.T)

        a3 = TrajectorySimulation(1000, 1000, probx=[.1, .1, .8], proby=[.3, .5, .2])
        d3 = list(a3.sample(1, 50, initpos=[d2['x'].iloc[-1], d2['y'].iloc[-1], d2['dirx'].iloc[-1], d2['diry'].iloc[-1]]))[0]
        v3 = infer_from_dataframe(d3, scale_numeric_types=True)
        t3 = JPT(variables=v3, targets=v3[2:], min_samples_leaf=int(d3.shape[0] * 0.1 / len(v3)))
        t3.learn(columns=d3.values.T)

        a4 = TrajectorySimulation(1000, 1000, probx=[.8, .1, .1], proby=[.2, .5, .3])
        d4 = list(a4.sample(1, 50, initpos=[d3['x'].iloc[-1], d3['y'].iloc[-1], d3['dirx'].iloc[-1], d3['diry'].iloc[-1]]))[0]
        v4 = infer_from_dataframe(d4, scale_numeric_types=True)
        t4 = JPT(variables=v4, targets=v4[2:], min_samples_leaf=int(d4.shape[0] * 0.1 / len(v4)))
        t4.learn(columns=d4.values.T)

        a5 = TrajectorySimulation(1000, 1000, probx=[.2, .5, .3], proby=[.5, .4, .1])
        d5 = list(a5.sample(1, 50, initpos=[d4['x'].iloc[-1], d4['y'].iloc[-1], d4['dirx'].iloc[-1], d4['diry'].iloc[-1]]))[0]
        v5 = infer_from_dataframe(d5, scale_numeric_types=True)
        t5 = JPT(variables=v5, targets=v5[2:], min_samples_leaf=int(d5.shape[0] * 0.1 / len(v5)))
        t5.learn(columns=d5.values.T)

        # JointProbabilityTreesMPE.trees.append((t1, v1, 'T1'))
        # JointProbabilityTreesMPE.trees.append((t2, v2, 'T2'))
        # JointProbabilityTreesMPE.trees.append((t3, v3, 'T3'))
        # JointProbabilityTreesMPE.trees.append((t4, v4, 'T4'))
        # JointProbabilityTreesMPE.trees.append((t5, v5, 'T5'))

        plt.plot(d1['x'], d1['y'], label='Trajectory 1')
        plt.scatter(d1['x'].iloc[0], d1['y'].iloc[0], label='Start 1', c='k')
        plt.scatter(d1['x'].iloc[-1], d1['y'].iloc[-1], label='End 1')

        plt.plot(d2['x'], d2['y'], label='Trajectory 2')
        plt.scatter(d2['x'].iloc[0], d2['y'].iloc[0], label='Start 2', c='k')
        plt.scatter(d2['x'].iloc[-1], d2['y'].iloc[-1], label='End 2')

        plt.plot(d3['x'], d3['y'], label='Trajectory 3')
        plt.scatter(d3['x'].iloc[0], d3['y'].iloc[0], label='Start 3', c='k')
        plt.scatter(d3['x'].iloc[-1], d3['y'].iloc[-1], label='End 3')

        plt.plot(d4['x'], d4['y'], label='Trajectory 4')
        plt.scatter(d4['x'].iloc[0], d4['y'].iloc[0], label='Start 4', c='k')
        plt.scatter(d4['x'].iloc[-1], d4['y'].iloc[-1], label='End 4')

        plt.plot(d5['x'], d5['y'], label='Trajectory 5')
        plt.scatter(d5['x'].iloc[0], d5['y'].iloc[0], label='Start 5', c='k')
        plt.scatter(d5['x'].iloc[-1], d5['y'].iloc[-1], label='End 5')

        # plt.ylim(0, 100)
        # plt.xlim(0, 100)
        plt.grid()
        plt.legend()
        plt.show()

        # t1.plot(title=f'Tree 1', plotvars=v1)
        # t2.plot(title=f'Tree 2', plotvars=v2)
        # t3.plot(title=f'Tree 3', plotvars=v3)
        # t4.plot(title=f'Tree 4', plotvars=v4)
        # t5.plot(title=f'Tree 5', plotvars=v5)

    def test_robot_multi_action_multi_sample(self) -> None:
        a1 = TrajectorySimulation(1000, 1000, probx=[.39, .01, .6], proby=[.59, .01, .4])
        a2 = TrajectorySimulation(1000, 1000, probx=[.6, .1, .3], proby=[.2, .1, .7])
        a3 = TrajectorySimulation(1000, 1000, probx=[.1, .1, .8], proby=[.3, .5, .2])
        a4 = TrajectorySimulation(1000, 1000, probx=[.8, .1, .1], proby=[.2, .5, .3])
        a5 = TrajectorySimulation(1000, 1000, probx=[.2, .5, .3], proby=[.5, .4, .1])


        d1 = pd.DataFrame(a1.sample(50, initpos=[500, 500, 'no', 'no']), columns=['x', 'y', 'dirx', 'diry'])
        v1 = infer_from_dataframe(d1, scale_numeric_types=True)
        t1 = JPT(variables=v1, targets=v1[2:], min_samples_leaf=int(d1.shape[0] * 0.1 / len(v1)))
        t1.learn(columns=d1.values.T)

        d2 = pd.DataFrame(a2.sample(50, initpos=[d1['x'].iloc[-1], d1['y'].iloc[-1], d1['dirx'].iloc[-1], d1['diry'].iloc[-1]]), columns=['x', 'y', 'dirx', 'diry'])
        v2 = infer_from_dataframe(d2, scale_numeric_types=True)
        t2 = JPT(variables=v2, targets=v2[2:], min_samples_leaf=int(d2.shape[0] * 0.1 / len(v2)))
        t2.learn(columns=d2.values.T)

        d3 = pd.DataFrame(a3.sample(50, initpos=[d2['x'].iloc[-1], d2['y'].iloc[-1], d2['dirx'].iloc[-1], d2['diry'].iloc[-1]]), columns=['x', 'y', 'dirx', 'diry'])
        v3 = infer_from_dataframe(d3, scale_numeric_types=True)
        t3 = JPT(variables=v3, targets=v3[2:], min_samples_leaf=int(d3.shape[0] * 0.1 / len(v3)))
        t3.learn(columns=d3.values.T)

        d4 = pd.DataFrame(a4.sample(50, initpos=[d3['x'].iloc[-1], d3['y'].iloc[-1], d3['dirx'].iloc[-1], d3['diry'].iloc[-1]]), columns=['x', 'y', 'dirx', 'diry'])
        v4 = infer_from_dataframe(d4, scale_numeric_types=True)
        t4 = JPT(variables=v4, targets=v4[2:], min_samples_leaf=int(d4.shape[0] * 0.1 / len(v4)))
        t4.learn(columns=d4.values.T)

        d5 = pd.DataFrame(a5.sample(50, initpos=[d4['x'].iloc[-1], d4['y'].iloc[-1], d4['dirx'].iloc[-1], d4['diry'].iloc[-1]]), columns=['x', 'y', 'dirx', 'diry'])
        v5 = infer_from_dataframe(d5, scale_numeric_types=True)
        t5 = JPT(variables=v5, targets=v5[2:], min_samples_leaf=int(d5.shape[0] * 0.1 / len(v5)))
        t5.learn(columns=d5.values.T)

        plt.plot(d1['x'], d1['y'], label='Trajectory 1')
        plt.scatter(d1['x'].iloc[0], d1['y'].iloc[0], label='Start 1')
        plt.scatter(d1['x'].iloc[-1], d1['y'].iloc[-1], label='End 1')

        plt.plot(d2['x'], d2['y'], label='Trajectory 2')
        plt.scatter(d2['x'].iloc[0], d2['y'].iloc[0], label='Start 2')
        plt.scatter(d2['x'].iloc[-1], d2['y'].iloc[-1], label='End 2')

        plt.plot(d3['x'], d3['y'], label='Trajectory 3')
        plt.scatter(d3['x'].iloc[0], d3['y'].iloc[0], label='Start 3')
        plt.scatter(d3['x'].iloc[-3], d3['y'].iloc[-3], label='End 3')

        plt.plot(d4['x'], d4['y'], label='Trajectory 4')
        plt.scatter(d4['x'].iloc[0], d4['y'].iloc[0], label='Start 4')
        plt.scatter(d4['x'].iloc[-4], d4['y'].iloc[-4], label='End 4')

        plt.plot(d5['x'], d5['y'], label='Trajectory 5')
        plt.scatter(d5['x'].iloc[0], d5['y'].iloc[0], label='Start 5')
        plt.scatter(d5['x'].iloc[-5], d5['y'].iloc[-5], label='End 5')

    def test_robot_single_action_multi_sample(self) -> None:
        a1 = TrajectorySimulation(1000, 1000, probx=[.39, .01, .6], proby=[.59, .01, .4])
        d1_ = list(a1.sample(20, 500, initpos=[500, 500, 'no', 'no']))
        d1 = pd.concat(d1_)

        for i, d in enumerate(d1_):
            plt.plot(d['x'], d['y'], label=f'Trajectory {i}')
            # plt.scatter(d['x'], d['y'], c='gray')
            plt.scatter(d['x'].iloc[0], d['y'].iloc[0], label=f'Start {i}', c='k')
            plt.scatter(d['x'].iloc[-1], d['y'].iloc[-1], label=f'End {i}')

        # plt.xlim(0, 1000)
        # plt.ylim(0, 1000)
        plt.grid()
        plt.legend()
        plt.show()

        # v1 = infer_from_dataframe(d1, scale_numeric_types=True)
        # t1 = JPT(variables=v1, targets=v1[2:], min_samples_leaf=int(d1.shape[0] * 0.1 / len(v1)))
        # t1.learn(columns=d1.values.T)

    def test_gaussian(self):
        gauss1 = Gaussian([5, 1], [[2, .7], [.7, .5]])
        gauss2 = Gaussian([5, 1], [[2, 1], [1, .5]])
        gauss3 = Gaussian([5, 1], [[2, -.7], [-.7, .3]])
        gauss4 = Gaussian([0, 0], [[1, -1], [-1, 1]])

        data1 = gauss1.sample(500)
        data2 = gauss2.sample(500)
        data3 = gauss3.sample(500)
        data4 = gauss4.sample(500)
        plt.scatter(data1[:, 0], data1[:, 1], marker='*')
        # plt.scatter(data2[:, 0], data2[:, 1], marker='x')
        plt.scatter(data3[:, 0], data3[:, 1], marker='^')
        # plt.scatter(data4[:, 0], data4[:, 1], marker='^')

        plt.grid()
        plt.legend()
        plt.show()

    def test_robot_pos(self):
        dt = f'{datetime.datetime.now().strftime(FILESTRFMT)}'

        # write sample data for MOVEFORWARD and TURN action of robot (absolute positions)
        for j in range(100):
            poses = []  # for plotting
            turns = []

            # init agent
            a = Agent([0, 0], [1, 0])

            turns.append(a.dir + (-90,))
            Move.turnleft(a)

            poses.append(a.pos+a.dir+(5,))
            Move.moveforward(a, 5)

            turns.append(a.dir + (5,))
            Move.turndeg(a, 5)

            poses.append(a.pos+a.dir+(2,))
            Move.moveforward(a, 2)

            turns.append(a.dir + (5,))
            Move.turndeg(a, 5)

            poses.append(a.pos+a.dir+(10,))
            Move.moveforward(a, 10)

            turns.append(a.dir + (90,))
            Move.turnright(a)

            poses.append(a.pos+a.dir+(10,))
            Move.moveforward(a, 10)

            turns.append(a.dir + (-5,))
            Move.turndeg(a, -5)

            poses.append(a.pos+a.dir+(10,))
            Move.moveforward(a, 10)

            turns.append(a.dir + (-90,))
            Move.turnleft(a)

            poses.append(a.pos+a.dir+(5,))
            Move.moveforward(a, 5)

            turns.append(a.dir + (5,))
            Move.turndeg(a, 5)

            poses.append(a.pos+a.dir+(2,))
            Move.moveforward(a, 2)

            turns.append(a.dir + (5,))
            Move.turndeg(a, 5)

            poses.append(a.pos+a.dir+(10,))
            Move.moveforward(a, 10)

            turns.append(a.dir + (90,))
            Move.turnright(a)

            poses.append(a.pos+a.dir+(10,))
            Move.moveforward(a, 10)

            turns.append(a.dir + (-5,))
            Move.turndeg(a, -5)

            poses.append(a.pos+a.dir+(10,))
            Move.moveforward(a, 10)

            poses.append(a.pos+a.dir+(0,))
            turns.append(a.dir + (0,))

            df_moveforward = pd.DataFrame(poses, columns=['x', 'y', 'xdir', 'ydir', 'numsteps'])
            df_moveforward.to_csv(os.path.join(locs.logs, f'{dt}-{j}-MOVEFORWARD.csv'), index=False)
            
            df_turn = pd.DataFrame(turns, columns=['xdir', 'ydir', 'angle'])
            df_turn.to_csv(os.path.join(locs.logs, f'{dt}-{j}-TURN.csv'), index=False)

            plt.scatter(df_moveforward['x'], df_moveforward['y'], marker='*', c='cornflowerblue')
            plt.plot(df_moveforward['x'], df_moveforward['y'], c='cornflowerblue')
            plt.scatter(df_moveforward['x'].iloc[0], df_moveforward['y'].iloc[0], marker='o', c='green', label='Start')
            plt.scatter(df_moveforward['x'].iloc[-1], df_moveforward['y'].iloc[-1], marker='o', c='red', label='End')
            plt.savefig(os.path.join(locs.logs, f'{dt}-{j}-MOVEFORWARD.png'))

        plt.grid()
        plt.legend()
        plt.show()


    def test_calo(self):
        calo = CALO()
        calo.adddatapath(locs.logs)
        calo.reloadmodels()

        # jpt = JPT.load(os.path.join(locs.logs, '2022-08-04_09:59-ALL-MOVEFORWARD.tree'))
        JointProbabilityTreesMPE.dt = '2022-08-19_12:15'
        jpt = calo.models[f'{JointProbabilityTreesMPE.dt}-ALL-MOVEFORWARD.tree']
        deltax = jpt.variables[0]
        deltay = jpt.variables[1]
        q = {deltax: [0, 1]}
        # q = {deltax: [0, 1], deltay: [0, 2]}
        r = jpt.reverse(q)
        print('Query:', q, 'result:', pprint.pformat(r))

    def test_robot_trajectory(self):
        a = Agent([0, 0], [1, 0])
        p = [.2, .1, .1, .6]
        actions = [Move.turnleft, Move.turnright, Move.turndeg, Move.moveforward]
        poses = Move.sampletrajectory(a, actions=actions, p=p, steps=100)
        print(poses)
