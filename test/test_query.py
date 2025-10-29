import logging
import os, sys
from typing import List
from unittest import TestCase

import numpy as np
from ddt import data, ddt

from bayrob.core.astar_jpt import Goal
from bayrob.core.base import BayRoB, Search
from bayrob.utils import locs
from bayrob.utils.constants import querypresets, searchpresets, obstacle_kitchen_boundaries, obstacles
from bayrob.utils.plotlib import plot_path, gendata, plot_pos, plot_dir, fig_to_file, defaultconfig
from bayrob.utils.utils import fmt

@ddt
class BayRoBQueryTest(TestCase):
    BAYROB = None
    PRESET_QUERY_VARS = None
    PRESET_QUERY_JPT = querypresets
    PRESET_SEARCH = searchpresets

    def setUp(self):
        self.BAYROB = BayRoB()
        self.BAYROB.adddatapath(
            [os.path.join(locs.examples, 'demo', d) for d in os.listdir(os.path.join(locs.examples, 'demo'))]
        )
        self.PRESET_QUERY_VARS = {
            'perception': {
                f'x_in x y_in': [self.BAYROB.models["perception"].varnames["x_in"],
                                 self.BAYROB.models["perception"].varnames["y_in"]],
                f'xdir_in x ydir_in': [self.BAYROB.models["perception"].varnames["xdir_in"],
                                       self.BAYROB.models["perception"].varnames["ydir_in"]],
                **{
                    k: v for k, v in self.BAYROB.models['perception'].varnames.items() if
                    k not in ['x_in', 'y_in', 'xdir_in', 'ydir_in']
                }
            },
            'move': {
                f'x_in x y_in': [self.BAYROB.models["move"].varnames["x_in"],
                                 self.BAYROB.models["move"].varnames["y_in"]],
                f'xdir_in x ydir_in': [self.BAYROB.models["move"].varnames["xdir_in"],
                                       self.BAYROB.models["move"].varnames["ydir_in"]],
                f'x_out x y_out': [self.BAYROB.models["move"].varnames["x_in"],
                                   self.BAYROB.models["move"].varnames["y_in"]],
                **{
                    k: v for k, v in self.BAYROB.models['move'].varnames.items() if
                    k not in ['x_in', 'y_in', 'xdir_in', 'ydir_in', 'x_out', 'y_out']
                }
            },
            'turn': {
                f'xdir_in x ydir_in': [self.BAYROB.models["turn"].varnames["xdir_in"],
                                       self.BAYROB.models["turn"].varnames["ydir_in"]],
                f'xdir_out x ydir_out': [self.BAYROB.models["turn"].varnames["xdir_out"],
                                         self.BAYROB.models["turn"].varnames["ydir_out"]],
                **{
                    k: v for k, v in self.BAYROB.models['turn'].varnames.items() if
                    k not in ['xdir_in', 'ydir_in', 'xdir_out', 'ydir_out']
                }
            },
            'pr2': {
                f't_x x t_y': [self.BAYROB.models["pr2"].varnames["t_x"],
                               self.BAYROB.models["pr2"].varnames["t_y"]],
                **{
                    k: v for k, v in self.BAYROB.models['pr2'].varnames.items() if
                    k not in ['t_x', 't_y', 't_z', 'duration', 'angle_z']
                }
            },
            'alarm': {
                k: v for k, v in self.BAYROB.models['alarm'].varnames.items()
            }
        }

        self.allvars = self.BAYROB.models['move'].variables + self.BAYROB.models['turn'].variables + \
                  self.BAYROB.models['perception'].variables
        self.allvars_ = {v.name: v for v in self.allvars}

    # ------------------------------------------------------------------------------------------------------------------

    def gendata_path(
            self,
            xvar: str,
            yvar: str,
            path: List
    ):
        # generate data for path plot (result of search)
        d = [
            (
                np.mean([s[xvar].mpe()[0].lower, s[xvar].mpe()[0].upper]),  # x
                np.mean([s[yvar].mpe()[0].lower, s[yvar].mpe()[0].upper]),  # y
                np.mean([s['xdir_in'].mpe()[0].lower, s['xdir_in'].mpe()[0].upper]),  # dx
                np.mean([s['ydir_in'].mpe()[0].lower, s['ydir_in'].mpe()[0].upper]),  # dy
                f'Step {i}',  # step
                f'<b>Step {i}</b><br>'
                f'<b>{"ROOT" if s.leaf is None or s.tree is None else f"{s.tree}-Leaf#{s.leaf}"}</b><br>'
                f'<b>MPEs:</b><br>'
                f'{"<br>".join(f"<i>{k}:</i> {fmt(v)}" for k, v in s.items())}<br>'
                f'<b>Expectations:</b><br>'
                f'{"<br>".join(f"<i>{k}:</i> {fmt(v.expectation())}" for k, v in s.items())}<br>',
                1  # size
            )
            for i, s in enumerate(path) if not isinstance(s, Goal)
        ]

        return d

    # ------------------------------------------------------------------------------------------------------------------

    @data("short", "multinomial")
    # @data("short")
    def test_search(self, p):
        # Arrange
        bayrob = self.BAYROB
        preset = self.PRESET_SEARCH[p]

        search = Search()
        search.init = {self.allvars_[k]: v for k, v in preset['init'].items()}
        search.init_tolerances = {self.allvars_[k]: v for k, v in preset['init_tolerances'].items()}
        search.goal = {self.allvars_[k]: v for k, v in preset['goal'].items()}
        search.goal_tolerances = {self.allvars_[k]: v for k, v in preset['goal_tolerances'].items()}
        search.bwd = preset['bwd']

        bayrob.query = search
        bayrob.runfunction = 'astar'

        print(f"Models count: {len(bayrob.models)}")
        print(f"Multiprocessing enabled: {bayrob.result.astar.use_multiprocessing if hasattr(bayrob.result, 'astar') else 'Unknown'}")

        # Act
        bayrob.search_astar()
        result = bayrob.result
        path = result.result

        # Assert
        d = self.gendata_path(
            xvar='x_in',
            yvar='y_in',
            path=path
        )
        self.assertNotEqual(len(d), 0, "No data generated for path plot")
        if d:
            plot_path(
                xvar='x_in',
                yvar='y_in',
                p=path,
                d=d,
                obstacles=[obstacle_kitchen_boundaries] + obstacles
            )

        data_pos = [
            gendata(
                'x_in',
                'y_in',
                s,
                {},
            ) for i, s in enumerate(path) if not isinstance(s, Goal)
        ]
        self.assertNotEqual(len(data_pos), 0, "No data generated for position plot")
        if data_pos:
            plot_pos(
                path=path,
                d=data_pos,
                limx=(0, 100),
                limy=(0, 100)
            )

        data_dir = [
            gendata(
                'xdir_in',
                'ydir_in',
                s,
                {},
            ) for i, s in enumerate(path) if not isinstance(s, Goal)
        ]
        self.assertNotEqual(len(data_dir), 0, "No data generated for direction plot")
        if data_dir:
            plot_dir(
                path=path,
                d=data_dir,
                limx=(-3, 3),
                limy=(-3, 3)
            )

    # ------------------------------------------------------------------------------------------------------------------
