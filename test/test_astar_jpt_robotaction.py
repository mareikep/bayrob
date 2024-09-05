import os
import unittest
from pathlib import Path

import pandas as pd
from jpt.base.intervals import ContinuousSet

from bayrob.application.astar_jpt_app import SubAStar_, SubAStarBW_
from bayrob.core.astar import BiDirAStar, Node
from bayrob.core.astar_jpt import Goal, State
from bayrob.utils import locs
from bayrob.utils.constants import obstacles as obstacles_, obstacle_kitchen_boundaries, searchpresets
from bayrob.utils.plotlib import gendata, plot_heatmap, plot_path, defaultconfig, plot_pos, plotly_animation, plot_dir
from bayrob.utils.utils import recent_example
from jpt import JPT, SymbolicVariable, SymbolicType
from jpt.distributions import Numeric, Gaussian, Bool
from jpt.distributions.quantile.quantiles import QuantileDistribution
from utils import uniform_numeric


class AStarRobotActionJPTTests(unittest.TestCase):

    # assumes trees robotaction_move and robotaction_move generated previously from functions in bayrob-dev/test/test_robotpos.py,
    # executed in the following order:
    # test_robot_pos_random (xor test_robot_pos) --> generates csv files from consecutive (random) move/turn actions
    # test_data_curation --> curates the previously generaed csv files to one large file each for robotaction_move and turn
    # actions; this is NOT just a merge of the files but a curation (initial data contains absolute positions or
    # directions, curated data contains deltas!)
    # test_jpt_robotaction_move --> learns JPT from robotaction_move data
    # test_jpt_turn --> learns JPT from turn data
    @classmethod
    def setUpClass(cls) -> None:
        cls.recent_move = recent_example(os.path.join(locs.examples, 'move'))
        cls.recent_turn = recent_example(os.path.join(locs.examples, 'turn'))
        cls.recent_perc = recent_example(os.path.join(locs.examples, 'perception'))
        print("loading examples from", cls.recent_move, cls.recent_turn, cls.recent_perc)

        cls.allobstacles = [obstacle_kitchen_boundaries] + obstacles_

        cls.models = dict(
            [
                (
                    Path(p).parent.name,
                    JPT.load(str(treefile))
                )
                for p in [cls.recent_move, cls.recent_turn, cls.recent_perc]
                for treefile in Path(p).glob('*.tree')
            ]
        )

        print('Loaded models', cls.models)
        # # plot initial distributions over x/y positions
        # t = cls.models['move']
        # # plot_tree_dist(
        # #     tree=t,
        # #     qvarx=t.varnames['x_in'],
        # #     qvary=t.varnames['y_in'],
        # #     title='Initial distribution P(x,y)',
        # #     limx=(0, 100),
        # #     limy=(0, 100),
        # #     save=os.path.join(os.path.join(recent, 'plots', '000-init-dist.html')),
        # #     show=True
        # # )
        #
        # initx, inity, initdirx, initdiry = [3.5, 58.5, .75, .75] #  [3, 60, 1, 0]
        # goalx, goaly = [6, 60]
        # tolerance_pos = 0.05
        # tolerance_dir = .01
        #
        # dx = Gaussian(initx, tolerance_pos).sample(500)
        # distx = Numeric()
        # distx.fit(dx.reshape(-1, 1), col=0)
        #
        # dy = Gaussian(inity, tolerance_pos).sample(500)
        # disty = Numeric()
        # disty.fit(dy.reshape(-1, 1), col=0)
        #
        # ddx = Gaussian(initdirx, tolerance_dir).sample(500)
        # distdx = Numeric()
        # distdx.fit(ddx.reshape(-1, 1), col=0)
        #
        # ddy = Gaussian(initdiry, tolerance_dir).sample(500)
        # distdy = Numeric()
        # distdy.fit(ddy.reshape(-1, 1), col=0)
        #
        # cls.init = State(
        #     {
        #         'x_in': distx,
        #         'y_in': disty,
        #         'xdir_in': distdx,
        #         'ydir_in': distdy
        #     }
        # )
        # # cls.initstate.plot(show=True)
        #
        # tol = .5
        # cls.goal = Goal(
        #     {
        #         'x_in': ContinuousSet(goalx - tol, goalx + tol),
        #         'y_in': ContinuousSet(goaly - tol, goaly + tol)
        #     }
        # )

    def pathexecution(self, initstate, cmds, shift=False):
        s = initstate
        p = [[s, {}]]
        for i, cmd in enumerate(cmds):
            print(f'Step {i} of {len(cmds)}: {repr(cmd)}')
            if cmd.tree is None:
                continue
            t = cmd.tree
            best = self.models[t].leaves[cmd.leaf]

            if best is None:
                print('skipping command', str(cmd), 'unsatisfiable!')
                continue

            # create successor state
            s_ = State()
            s_.update({k: v for k, v in s.items()})
            s_.tree = cmd.tree
            s_.leaf = cmd.leaf

            # update belief state of potential predecessor
            nsegments = 20
            for vn, d in best.distributions.items():
                vname = vn.name
                outvar = vname.replace('_in', '_out')
                invar = vname.replace('_out', '_in')

                if vname.endswith('_out') and vname.replace('_out', '_in') in s_:
                    # if the _in variable is already contained in the state, update it by adding the delta
                    # from the leaf distribution
                    if shift:
                        s_[invar] = Numeric().set(
                            QuantileDistribution.from_cdf(s_[invar].cdf.xshift(-best.distributions[outvar].expectation())))
                    else:
                        indist = s_[invar]
                        outdist = best.distributions[outvar]
                        if len(indist.cdf.functions) > nsegments:
                            print(
                                f"A Approximating {invar} distribution of s_ with {len(indist.cdf.functions)} functions to {nsegments} functions")
                            indist = indist.approximate(n_segments=nsegments)
                            # s_[invar] = s_[invar].approximate_fast(eps=.01)
                        if len(outdist.cdf.functions) > nsegments:
                            print(
                                f"B Approximating {outvar} distribution of best with {len(outdist.cdf.functions)} functions to {nsegments} functions")
                            outdist = outdist.approximate(n_segments=nsegments)
                        vname = invar
                        s_[vname] = indist + outdist
                elif vname.endswith('_in') and vname in s_:
                    # do not overwrite '_in' distributions
                    continue
                else:
                    s_[vname] = d

                if not shift:
                    if hasattr(s_[vname], 'approximate'):
                        print(
                            f"C Approximating {vname} distribution of s_ (result) with {len(s_[vname].cdf.functions)} functions to {nsegments} functions")
                        s_[vname] = s_[vname].approximate(n_segments=nsegments)

            p.append([s_, {'tree': cmd.tree, 'leaf': cmd.leaf}])
            s = State()
            s.update({k: v for k, v in s_.items()})
        return p

    def plot_cram_path(self, p, plotpath=True, plotpos=False, plotcollision=False, plotdir=False):
        # plot annotated rectangles representing the obstacles and world boundaries

        if plotpath:
            # plot path as scatter points with direction arrows in kitchen world
            fig = plot_path(
                'x_in',
                'y_in',
                p,
                save=os.path.join(locs.logs, f'test_astar_cram_path.svg'),
                obstacles=self.allobstacles,
                show=False
            )

            fig.write_html(
                os.path.join(locs.logs, f'test_astar_cram_path.html'),
                config=defaultconfig(os.path.join(locs.logs, f'test_astar_cram_path.html')),
                include_plotlyjs="cdn"
            )

            fig.show(config=defaultconfig(os.path.join(locs.logs, f'test_astar_cram_path.html')))

        if plotpos:
            # plot animation of heatmap representing position distribution update
            plot_pos(
                path=p,
                save=os.path.join(locs.logs, f'test_astar_cram_path-animation.html'),
                show=True,
                limx=(0, 100),
                limy=(0, 100)
            )

            # plot animation of 3d surface representing position distribution update
            plot_pos(
                path=p,
                save=os.path.join(locs.logs, f'test_astar_cram_path-animation-3d.html'),
                show=True,
                limx=(0, 100),
                limy=(0, 100),
                fun="surface"
            )

        if plotcollision:
            # plot animation of collision bar chart representing change of collision status
            frames = [s['collided'].plot(view=False).data for (s, _) in p if 'collided' in s]
            plotly_animation(
                data=frames,
                save=os.path.join(locs.logs, f'collision.html'),
                show=True
            )

        if plotdir:
            plot_dir(
                path=p,
                save=os.path.join(locs.logs, f'test_astar_cram_path-dirxy.html'),
                show=True,
                limx=(-3, 3),
                limy=(-3, 3)
            )


    def test_isgoal_fw_t(self) -> None:
        g = Goal()
        g.update({
                'x_in': ContinuousSet(1.5, 4.5),
                'y_in': ContinuousSet(57.5, 62.5)
        })

        self.a_star = SubAStar_(
            self.init,
            g,
            models=self.models
        )

        n = Node(
            state=self.init,
            g=0,
            h=0,
            parent=None,
        )
        self.assertTrue(self.a_star.isgoal(n))

    def test_isgoal_fw_f(self) -> None:
        g = Goal()
        g.update({
            'x_in': ContinuousSet(70, 72),
            'y_in': ContinuousSet(50, 52)
        })

        self.a_star = SubAStar_(
            self.init,
            g,
            models=self.models
        )

        n = Node(
            state=self.init,
            g=0,
            h=0,
            parent=None,
        )

        self.assertFalse(self.a_star.isgoal(n))

    def test_h_fw(self) -> None:
        pass

    def test_stepcost_fw(self) -> None:
        pass

    def test_isgoal_bw(self) -> None:
        pass

    def test_h_bw(self) -> None:
        pass

    def test_stepcost_bw(self) -> None:
        pass

    def test_isgoal_bdir(self) -> None:
        pass

    def test_h_bir(self) -> None:
        pass

    def test_stepcost_bdir(self) -> None:
        pass

    # @unittest.skip
    def test_astar_fw_path_single(self) -> None:
        initx, inity, initdirx, initdiry = [3.5, 58.5, .75, .75]  # [3, 60, 1, 0]
        goalx, goaly = [5, 60]
        tolerance_pos = 0.05
        tolerance_dir = .01

        dx = Gaussian(initx, tolerance_pos).sample(500)
        distx = Numeric()
        distx.fit(dx.reshape(-1, 1), col=0)

        dy = Gaussian(inity, tolerance_pos).sample(500)
        disty = Numeric()
        disty.fit(dy.reshape(-1, 1), col=0)

        ddx = Gaussian(initdirx, tolerance_dir).sample(500)
        distdx = Numeric()
        distdx.fit(ddx.reshape(-1, 1), col=0)

        ddy = Gaussian(initdiry, tolerance_dir).sample(500)
        distdy = Numeric()
        distdy.fit(ddy.reshape(-1, 1), col=0)

        tol = .5
        dgx = Gaussian(goalx, tol).sample(500)
        distgx = Numeric()
        distgx.fit(dgx.reshape(-1, 1), col=0)

        dgy = Gaussian(goaly, tol).sample(500)
        distgy = Numeric()
        distgy.fit(dgy.reshape(-1, 1), col=0)
        goal = Goal(
            {
                'x_in': distgx,
                'y_in': distgy
            }
        )

        init = State(
            {
                'x_in': distx,
                'y_in': disty,
                'xdir_in': distdx,
                'ydir_in': distdy
            }
        )

        self.a_star = SubAStar_(
            init,
            goal,
            models=self.models
        )
        self.path = list(self.a_star.search())

        p = self.pathexecution(init, self.path, shift=False)
        self.plot_cram_path(p, plotpath=True, plotpos=True, plotdir=True)

    def test_astar_fw_path_multiple(self) -> None:
        initx, inity, initdirx, initdiry = [3.5, 58.5, .6, .3]  # [3, 60, 1, 0]
        goalx, goaly = [10, 65]
        tolerance_pos = 0.05
        tolerance_dir = .01

        dx = Gaussian(initx, tolerance_pos).sample(500)
        distx = Numeric()
        distx.fit(dx.reshape(-1, 1), col=0)

        dy = Gaussian(inity, tolerance_pos).sample(500)
        disty = Numeric()
        disty.fit(dy.reshape(-1, 1), col=0)

        ddx = Gaussian(initdirx, tolerance_dir).sample(500)
        distdx = Numeric()
        distdx.fit(ddx.reshape(-1, 1), col=0)

        ddy = Gaussian(initdiry, tolerance_dir).sample(500)
        distdy = Numeric()
        distdy.fit(ddy.reshape(-1, 1), col=0)

        tol = 1.5
        goal = Goal(
            {
                'x_in': ContinuousSet(goalx - tol, goalx + tol),
                'y_in': ContinuousSet(goaly - tol, goaly + tol)
            }
        )

        init = State(
            {
                'x_in': distx,
                'y_in': disty,
                'xdir_in': distdx,
                'ydir_in': distdy
            }
        )

        self.a_star = SubAStar_(
            init,
            goal,
            models=self.models
        )
        self.path = list(self.a_star.search())

    def test_astar_bw_path_single(self) -> None:
        initx, inity, initdirx, initdiry = [3.5, 57, .75, .75]  # [3, 60, 1, 0]
        goalx, goaly = [5, 60]
        tolerance_pos = 0.05
        tolerance_dir = .01

        dx = Gaussian(initx, tolerance_pos).sample(500)
        distx = Numeric()
        distx.fit(dx.reshape(-1, 1), col=0)

        dy = Gaussian(inity, tolerance_pos).sample(500)
        disty = Numeric()
        disty.fit(dy.reshape(-1, 1), col=0)

        ddx = Gaussian(initdirx, tolerance_dir).sample(500)
        distdx = Numeric()
        distdx.fit(ddx.reshape(-1, 1), col=0)

        ddy = Gaussian(initdiry, tolerance_dir).sample(500)
        distdy = Numeric()
        distdy.fit(ddy.reshape(-1, 1), col=0)

        tol = .5
        goal = Goal(
            {
                'x_in': ContinuousSet(goalx - tol, goalx + tol),
                'y_in': ContinuousSet(goaly - tol, goaly + tol)
            }
        )

        init = State(
            {
                'x_in': distx,
                'y_in': disty,
                'xdir_in': distdx,
                'ydir_in': distdy
            }
        )

        self.a_star = SubAStarBW_(
            init,
            goal,
            models=self.models
        )
        self.path = list(self.a_star.search())
        self.path.reverse()

        p = self.pathexecution(init, self.path, shift=False)
        self.plot_cram_path(p, plotpath=True, plotpos=True, plotdir=True)

    def test_astar_bw_path_multinomialgoal(self) -> None:
        initx, inity, initdirx, initdiry = [62, 74, .3, .9]  # <--- WORKS! DO NOT TOUCH
        # initx, inity, initdirx, initdiry = [62, 74, 1, 0]
        tolerance_pos = .1
        tolerance_dir = .01

        dx = Gaussian(initx, tolerance_pos).sample(500)
        distx = Numeric()
        distx.fit(dx.reshape(-1, 1), col=0)

        dy = Gaussian(inity, tolerance_pos).sample(500)
        disty = Numeric()
        disty.fit(dy.reshape(-1, 1), col=0)

        ddx = Gaussian(initdirx, tolerance_dir).sample(500)
        distdx = Numeric()
        distdx.fit(ddx.reshape(-1, 1), col=0)

        ddy = Gaussian(initdiry, tolerance_dir).sample(500)
        distdy = Numeric()
        distdy.fit(ddy.reshape(-1, 1), col=0)

        goal_ = {True}
        milk_ = self.models['perception'].varnames['detected(milk)'].distribution()
        dmilk = milk_.set([(1/len(goal_) if x in goal_ else 0) for x in list(milk_.values)])
        goal = Goal(
            {
                'detected(milk)': dmilk,
            }
        )

        init = State(
            {
                'x_in': distx,
                'y_in': disty,
                'xdir_in': distdx,
                'ydir_in': distdy
            }
        )

        models = dict(
            [
                (
                    Path(p).name,
                    JPT.load(str(treefile))
                )
                for p in [
                os.path.join(locs.examples, 'demo', 'move'),
                os.path.join(locs.examples, 'demo', 'turn'),
                os.path.join(locs.examples, 'demo', 'perception'),
                ]
                for treefile in Path(p).rglob('*.tree')
            ]
        )
        print(models)
        print(self.models)

        self.a_star = SubAStarBW_(
            init,
            goal,
            models=models
        )
        print(f"Running {self.a_star.__class__.__name__} with {repr(init)} and {goal}")
        self.path = list(self.a_star.search())
        self.path.reverse()

        # TEST: execution of found path, check if goal is met.

    def test_query_preset_example_session(self) -> None:
        from bayrob.core.base import BayRoB
        from bayrob.core.base import Query
        preset = {
            "evidence": {
                'detected(milk)': False,
                'x_in': ContinuousSet(58, 68),
                'y_in': ContinuousSet(70, 80),
                'nearest_furniture': 'fridge'
            },
            "queryvars": ['daytime', 'open(fridge_door)']
        }

        self.bayrob = BayRoB()
        self.bayrob.adddatapath([os.path.join(locs.examples, 'demo', "perception")])
        allvars_ = {v.name: v for v in self.bayrob.models['perception'].variables}

        qo = Query()
        qo.model = self.bayrob.models['perception']
        qo.evidence = {allvars_[k]: v for k, v in preset['evidence'].items()}
        qo.queryvars = [self.bayrob.models['perception'].varnames[k] for k in preset['queryvars']]

        self.bayrob.query = qo
        self.bayrob.query_jpts()
        print(self.bayrob.result.result)


    def test_astar_bw_path_multinomialgoal_session(self) -> None:
        from bayrob.core.base import BayRoB
        from bayrob.core.base import Search
        preset = searchpresets['multinomial']

        self.bayrob = BayRoB()
        self.bayrob.adddatapath([os.path.join(locs.examples, 'demo', d) for d in os.listdir(os.path.join(locs.examples, 'demo'))])
        allvars = self.bayrob.models['move'].variables + self.bayrob.models['turn'].variables + \
                  self.bayrob.models['perception'].variables
        allvars_ = {v.name: v for v in allvars}

        self.asr = Search()
        self.asr.bwd = preset['bwd']
        self.asr.init = {allvars_[k]: v for k, v in preset['init'].items()}
        self.asr.init_tolerances = {allvars_[k]: v for k, v in preset['init_tolerances'].items()}
        self.asr.goal = {allvars_[k]: v for k, v in preset['goal'].items()}
        self.asr.goal_tolerances = {allvars_[k]: v for k, v in preset['goal_tolerances'].items()}
        self.asr.bwd = preset['bwd']

        self.bayrob.query = self.asr
        self.bayrob.search_astar()
        print(self.bayrob.result)
        print(self.bayrob.result.result)

    def test_astar_bw_path_single_step_from_beliefstate(self) -> None:
        tp = self.models['perception']
        lp = tp.leaves[167]
        goal = Goal({k.name: v for k, v in lp.distributions.items()})
        goal.tree = 'perception'
        goal.leaf = lp.idx

        tm = self.models['move']
        lm = tm.leaves[666]
        init = State({k.name: v for k, v in lm.distributions.items() if k.name not in ['x_out', 'y_out', 'collided']})
        init.tree = 'move'
        init.leaf = lm.idx

        self.a_star = SubAStarBW_(
            init,
            goal,
            models=self.models
        )
        self.path = list(self.a_star.search())
        self.path.reverse()

    def test_astar_fw_path_multinomialgoal(self) -> None:
        initx, inity, initdirx, initdiry = [62, 72, .3, .9]  # <--- WORKS! DO NOT TOUCH
        # initx, inity, initdirx, initdiry = [62, 74, 1, 0]
        tolerance_pos = .1
        tolerance_dir = .01

        dx = Gaussian(initx, tolerance_pos).sample(500)
        distx = Numeric()
        distx.fit(dx.reshape(-1, 1), col=0)

        dy = Gaussian(inity, tolerance_pos).sample(500)
        disty = Numeric()
        disty.fit(dy.reshape(-1, 1), col=0)

        ddx = Gaussian(initdirx, tolerance_dir).sample(500)
        distdx = Numeric()
        distdx.fit(ddx.reshape(-1, 1), col=0)

        ddy = Gaussian(initdiry, tolerance_dir).sample(500)
        distdy = Numeric()
        distdy.fit(ddy.reshape(-1, 1), col=0)

        goal = Goal(
            {
                'detected(milk)': {True},
            }
        )

        init = State(
            {
                'x_in': distx,
                'y_in': disty,
                'xdir_in': distdx,
                'ydir_in': distdy
            }
        )

        self.a_star = SubAStar_(
            init,
            goal,
            models=self.models
        )
        self.path = list(self.a_star.search())

    # @unittest.skip
    def test_astar_bw_path_multiple(self) -> None:
        initx, inity, initdirx, initdiry = [3.5, 58.5, .3, .6]  # [3, 60, 1, 0]
        goalx, goaly = [10, 65]
        tolerance_pos = 0.05
        tolerance_dir = .01

        dx = Gaussian(initx, tolerance_pos).sample(500)
        distx = Numeric()
        distx.fit(dx.reshape(-1, 1), col=0)

        dy = Gaussian(inity, tolerance_pos).sample(500)
        disty = Numeric()
        disty.fit(dy.reshape(-1, 1), col=0)

        ddx = Gaussian(initdirx, tolerance_dir).sample(500)
        distdx = Numeric()
        distdx.fit(ddx.reshape(-1, 1), col=0)

        ddy = Gaussian(initdiry, tolerance_dir).sample(500)
        distdy = Numeric()
        distdy.fit(ddy.reshape(-1, 1), col=0)

        tol = 1.5
        goal = Goal(
            {
                'x_in': ContinuousSet(goalx - tol, goalx + tol),
                'y_in': ContinuousSet(goaly - tol, goaly + tol)
            }
        )

        init = State(
            {
                'x_in': distx,
                'y_in': disty,
                'xdir_in': distdx,
                'ydir_in': distdy
            }
        )

        self.a_star = SubAStarBW_(
            init,
            goal,
            models=self.models
        )
        self.path = list(self.a_star.search())
        self.path.reverse()

    @unittest.skip
    def test_astar_bdir_path(self) -> None:
        self.a_star = BiDirAStar(
            SubAStar_,
            SubAStarBW_,
            self.init,
            self.goal,
            models=self.models
        )

        self.path = list(self.a_star.search())

    def test_distance(self):
        # Arrange
        v1 = self.init['x_in']
        v2 = uniform_numeric(self.goal['x_in'].lower, self.goal['x_in'].upper)

        # Act
        res = Numeric.distance(v1, v2)

        # Assert
        self.assertAlmostEqual(res, 3., delta=.1)

    def test_distance_fw_multiple(self):
        # Arrange
        g = Goal()
        g.update({
            'v1': {True},
            'v2': {"green", "red", "blue"},
            'v3': {1, 2, 5},
            # 'v4': ContinuousSet(0, 3)
        })

        v1_ = SymbolicVariable('BoolVar', Bool)
        v1_ = v1_.distribution().set(2/3)

        v2_ = SymbolicType('MultiVar', labels=["green", "red", "blue", "yellow", "black", "orange"])().set([.3, .3, .1, .1, .1, .1])
        v3_ = SymbolicType('IntVar', labels=[1, 2, 3, 4, 5])().set([.2, .2, .1, .1, .4])

        v4 = Gaussian(1.3, .4).sample(500)
        v4_ = Numeric()
        v4_ = v4_.fit(v4.reshape(-1, 1), col=0)

        s1 = State()
        s1.update({
            'v1': v1_,
            'v2': v2_,
            'v3': v3_,
            'v4': v4_
        })

        # Act
        res = SubAStar_.dist(s1, g)

        # Assert
        self.assertAlmostEqual(res, .27, delta=.1)

    def test_distance_fw_bool_t(self):
        # Arrange
        g = Goal()
        g.update({
            'v1': {True},
        })

        v1_ = SymbolicVariable('BoolVar', Bool)
        v1_ = v1_.distribution().set(2/3)

        s1 = State()
        s1.update({
            'v1': v1_,
        })

        # Act
        res = SubAStar_.dist(s1, g)

        # Assert
        self.assertAlmostEqual(res, 1/3, 5)

    def test_distance_fw_bool_f(self):
        # Arrange
        g = Goal()
        g.update({
            'v1': {False},
        })

        v1_ = SymbolicVariable('BoolVar', Bool)
        v1_ = v1_.distribution().set(2/3)

        s1 = State()
        s1.update({
            'v1': v1_,
        })

        # Act
        res = SubAStar_.dist(s1, g)

        # Assert
        self.assertAlmostEqual(res, 2/3, 5)

    def test_distance_fw_multinomial_t(self):
        # Arrange
        g = Goal()
        g.update({
            'v2': {"green", "red", "blue"}
        })

        v2_ = SymbolicType('MultiVar', labels=["green", "red", "blue", "yellow", "black", "orange"])().set(
            [.2, .1, .1, .3, .1, .2])

        s1 = State()
        s1.update({
            'v2': v2_,
        })

        # Act
        res = SubAStar_.dist(s1, g)

        # Assert
        self.assertEqual(res, .6)

    def test_distance_fw_symmetric(self):
        # Arrange
        g = Goal()
        g.update({
            'v1': {True, False},
            'v2': {"green", "red", "blue"},
            'v3': {1, 2, 5},
            'v4': ContinuousSet(1, 3)
        })

        v1_ = SymbolicVariable('BoolVar', Bool)
        v1_ = v1_.distribution().set(2/3)

        v2_ = SymbolicType('MultiVar', labels=["green", "red", "blue", "yellow", "black", "orange"])().set([.2, .1, .1, .3, .1, .2])
        v3_ = SymbolicType('IntVar', labels=[1, 2, 3, 4, 5])().set([.2, .2, .1, .1, .4])

        v4 = Gaussian(1, .2).sample(500)
        v4_ = Numeric()
        v4_ = v4_.fit(v4.reshape(-1, 1), col=0)

        v5 = Gaussian(5, .2).sample(500)
        v5_ = Numeric()
        v5_ = v5_.fit(v5.reshape(-1, 1), col=0)

        s1 = State()
        s1.update({
            'v1': v1_,
            # 'v2': v2_,
            # 'v3': v3_,
            'v4': v4_
        })

        # Act
        res1 = SubAStar_.dist(s1, g)
        res2 = SubAStar_.dist(g, s1)

        # Assert
        self.assertEqual(res1, res2)

    def test_distance_fw_numeric(self):
        # Arrange
        g = Goal()
        g.update({
            'v4': ContinuousSet(1, 3)
        })

        v4 = Gaussian(1, .2).sample(500)
        v4_ = Numeric()
        v4_ = v4_.fit(v4.reshape(-1, 1), col=0)

        v5 = Gaussian(5, .2).sample(500)
        v5_ = Numeric()
        v5_ = v5_.fit(v5.reshape(-1, 1), col=0)

        s1 = State()
        s1.update({
            'v4': v4_
        })

        # Act
        res1 = SubAStar_.dist(s1, g)

        # Assert
        self.assertAlmostEqual(0.98, res1, delta=.1)

    def test_distance_bw_multiple(self):
        # Arrange
        g = Goal()
        g.update({
            'v1': {True},
            'v2': {"green", "red", "blue"},
            'v3': {1, 2, 5},
            'v4': ContinuousSet(0, 3)
        })

        v1_ = SymbolicVariable('BoolVar', Bool)
        v1_ = v1_.distribution().set(2/3)

        v2_ = SymbolicType('MultiVar', labels=["green", "red", "blue", "yellow", "black", "orange"])().set([.3, .3, .1, .1, .1, .1])
        v3_ = SymbolicType('IntVar', labels=[1, 2, 3, 4, 5])().set([.2, .2, .1, .1, .4])

        v4 = Gaussian(1.3, .4).sample(500)
        v4_ = Numeric()
        v4_ = v4_.fit(v4.reshape(-1, 1), col=0)

        s1 = State()
        s1.update({
            'v1': v1_,
            'v2': v2_,
            'v3': v3_,
            'v4': v4_
        })

        # Act
        res = SubAStarBW_.dist(s1, g)

        # Assert
        self.assertAlmostEqual(res, .27, delta=.02)

    def test_distance_bw_bool_t(self):
        # Arrange
        g = Goal()
        g.update({
            'v1': {True},
        })

        v1_ = SymbolicVariable('BoolVar', Bool)
        v1_ = v1_.distribution().set(2/3)

        s1 = State()
        s1.update({
            'v1': v1_,
        })

        # Act
        res = SubAStar_.dist(s1, g)

        # Assert
        self.assertAlmostEqual(res, 1/3, 5)

    def test_distance_bw_bool_f(self):
        # Arrange
        g = Goal()
        g.update({
            'v1': {False},
        })

        v1_ = SymbolicVariable('BoolVar', Bool)
        v1_ = v1_.distribution().set(2/3)

        s1 = State()
        s1.update({
            'v1': v1_,
        })

        # Act
        res = SubAStarBW_.dist(s1, g)

        # Assert
        self.assertAlmostEqual(res, 2/3, 5)

    def test_distance_bw_multinomial_t(self):
        # Arrange
        g = Goal()
        g.update({
            'v2': {"green", "red", "blue"}
        })

        v2_ = SymbolicType('MultiVar', labels=["green", "red", "blue", "yellow", "black", "orange"])().set(
            [.2, .1, .1, .3, .1, .2])

        s1 = State()
        s1.update({
            'v2': v2_,
        })

        # Act
        res = SubAStarBW_.dist(s1, g)

        # Assert
        self.assertEqual(res, .6)

    def test_distance_bw_symmetric(self):
        # Arrange
        g = Goal()
        g.update({
            'v1': {True, False},
            'v2': {"green", "red", "blue"},
            'v3': {1, 2, 5},
            'v4': ContinuousSet(1, 3)
        })

        v1_ = SymbolicVariable('BoolVar', Bool)
        v1_ = v1_.distribution().set(2/3)

        v2_ = SymbolicType('MultiVar', labels=["green", "red", "blue", "yellow", "black", "orange"])().set([.2, .1, .1, .3, .1, .2])
        v3_ = SymbolicType('IntVar', labels=[1, 2, 3, 4, 5])().set([.2, .2, .1, .1, .4])

        v4 = Gaussian(1, .2).sample(500)
        v4_ = Numeric()
        v4_ = v4_.fit(v4.reshape(-1, 1), col=0)

        v5 = Gaussian(5, .2).sample(500)
        v5_ = Numeric()
        v5_ = v5_.fit(v5.reshape(-1, 1), col=0)

        s1 = State()
        s1.update({
            'v1': v1_,
            # 'v2': v2_,
            # 'v3': v3_,
            'v4': v4_
        })

        # Act
        res1 = SubAStarBW_.dist(s1, g)
        res2 = SubAStarBW_.dist(g, s1)

        # Assert
        self.assertEqual(res1, res2)

    def test_distance_bw_numeric(self):
        # Arrange
        g = Goal({'v4': ContinuousSet(1, 3)})

        v4 = Gaussian(1, .2).sample(500)
        v4_ = Numeric()
        v4_ = v4_.fit(v4.reshape(-1, 1), col=0)

        v5 = Gaussian(5, .2).sample(500)
        v5_ = Numeric()
        v5_ = v5_.fit(v5.reshape(-1, 1), col=0)

        # Act
        res1 = SubAStarBW_.dist(State({'v4': v4_}), g)
        res2 = SubAStarBW_.dist(State({'v4': v5_}), g)

        # Assert
        self.assertAlmostEqual(res1, 0.98, delta=.2)
        self.assertAlmostEqual(res2, 3, delta=.2)

    def test_isgoal_fw_numeric_clear_cases(self):
        # Arrange
        g = Goal()
        g.update({
            'v4': ContinuousSet(1, 3)
        })

        v4 = Gaussian(2, .2).sample(500)
        v4_ = Numeric()
        v4_ = v4_.fit(v4.reshape(-1, 1), col=0)

        v5 = Gaussian(5, .2).sample(500)
        v5_ = Numeric()
        v5_ = v5_.fit(v5.reshape(-1, 1), col=0)

        s1 = State()
        s1.update({
            'v4': v4_
        })

        s2 = State()
        s2.update({
            'v4': v5_
        })

        astar1 = SubAStar_(
            s1,
            g,
            models=self.models
        )

        astar2 = SubAStar_(
            s1,
            g,
            models=self.models
        )

        # Act
        res1 = astar1.isgoal(Node(state=s1, g=0, h=0))
        res2 = astar2.isgoal(Node(state=s2, g=0, h=0))

        # Assert
        self.assertTrue(res1)
        self.assertFalse(res2)

    def test_isgoal_fw_numeric_margins(self):
        # Arrange
        v4 = Gaussian(1.3, .2).sample(500)
        v4_ = Numeric()
        v4_ = v4_.fit(v4.reshape(-1, 1), col=0)

        v5 = Gaussian(2.8, .5).sample(500)
        v5_ = Numeric()
        v5_ = v5_.fit(v5.reshape(-1, 1), col=0)

        s1 = State({'v4': v4_})
        s2 = State({'v4': v5_})

        astar1 = SubAStar_(
            s1,
            Goal({'v4': ContinuousSet(1, 3)}),
            models=self.models
        )

        # Act
        res1 = astar1.isgoal(Node(state=s1, g=0, h=0))
        res2 = astar1.isgoal(Node(state=s2, g=0, h=0))

        # Assert
        self.assertTrue(res1)
        self.assertFalse(res2)

    def test_isgoal_fw_multinomial(self):
        # Arrange
        g = Goal()
        g.update({
            'v2': {"red", "green", "blue"}
        })

        v2_ = SymbolicType('MultiVar', labels=["green", "red", "blue", "yellow", "black", "orange"])().set([.3, .3, .1, .1, .1, .1])
        v3_ = SymbolicType('MultiVar', labels=["green", "red", "blue", "yellow", "black", "orange"])().set([.2, .1, .1, .3, .1, .2])

        s1 = State()
        s1.update({
            'v2': v2_
        })

        s2 = State()
        s2.update({
            'v2': v3_
        })

        astar1 = SubAStar_(
            s1,
            Goal({
                'v2': {"red", "green", "blue"}
            }),
            models=self.models
        )

        astar2 = SubAStar_(
            s1,
            g,
            models=self.models
        )

        # Act
        res1 = astar1.isgoal(Node(state=s1, g=0, h=0))
        res2 = astar2.isgoal(Node(state=s2, g=0, h=0))

        # Assert
        self.assertTrue(res1)
        self.assertFalse(res2)

    def test_isgoal_fw_multiple(self):
        # Arrange
        g = Goal()
        g.update({
            'v1': {True},
            'v2': {"green", "red", "blue"},
            'v3': {1, 2, 5},
            'v4': ContinuousSet(1, 3)
        })

        v1_ = SymbolicVariable('BoolVar', Bool)
        v1_ = v1_.distribution().set(2/3)

        v2_ = SymbolicType('MultiVar', labels=["green", "red", "blue", "yellow", "black", "orange"])().set([.3, .3, .1, .1, .1, .1])
        v3_ = SymbolicType('IntVar', labels=[1, 2, 3, 4, 5])().set([.2, .2, .1, .1, .4])

        v4 = Gaussian(1.3, .4).sample(500)
        v4_ = Numeric()
        v4_ = v4_.fit(v4.reshape(-1, 1), col=0)

        s1 = State()
        s1.update({
            'v1': v1_,
            'v2': v2_,
            'v3': v3_,
            'v4': v4_
        })

        astar1 = SubAStar_(
            s1,
            g,
            models=self.models
        )

        # Act
        res1 = astar1.isgoal(Node(state=s1, g=0, h=0))

        # Assert
        self.assertTrue(res1)

    def test_isgoal_bw_numeric_init_is_goal(self):
        # Arrange
        i = Gaussian(2, .2).sample(500)
        i_ = Numeric()
        i_ = i_.fit(i.reshape(-1, 1), col=0)

        v4 = Gaussian(2, .2).sample(500)
        v4_ = Numeric()
        v4_ = v4_.fit(v4.reshape(-1, 1), col=0)

        astar1 = SubAStarBW_(
            State({'v4': i_}),
            Goal({'v4': ContinuousSet(1, 3)}),
            models=self.models
        )

        # Act
        res1 = astar1.isgoal(Node(state=Goal({'v4': ContinuousSet(1, 3)}), g=0, h=0))
        res2 = astar1.isgoal(Node(state=Goal({'v4': ContinuousSet(3, 4)}), g=0, h=0))
        res3 = astar1.isgoal(Node(state=State({'v4': v4_}), g=0, h=0))

        # Assert
        self.assertTrue(res1)
        self.assertFalse(res2)
        self.assertTrue(res3)

    def test_isgoal_bw_numeric_clear_cases(self):
        # Arrange
        i = Gaussian(2, .2).sample(500)
        i_ = Numeric()
        i_ = i_.fit(i.reshape(-1, 1), col=0)

        v4 = Gaussian(2, .2).sample(500)
        v4_ = Numeric()
        v4_ = v4_.fit(v4.reshape(-1, 1), col=0)

        v5 = Gaussian(5, .2).sample(500)
        v5_ = Numeric()
        v5_ = v5_.fit(v5.reshape(-1, 1), col=0)

        v6 = Gaussian(2, 1).sample(500)
        v6_ = Numeric()
        v6_ = v6_.fit(v6.reshape(-1, 1), col=0)

        astar1 = SubAStarBW_(
            State({'v4': i_}),
            Goal({'v4': ContinuousSet(1, 3)}),
            models=self.models
        )

        # Act
        res1 = astar1.isgoal(Node(state=State({'v4': i_}), g=0, h=0))
        res2 = astar1.isgoal(Node(state=State({'v4': v4_}), g=0, h=0))
        res3 = astar1.isgoal(Node(state=State({'v4': v5_}), g=0, h=0))
        res4 = astar1.isgoal(Node(state=State({'v4': v6_}), g=0, h=0))

        # Assert
        self.assertTrue(res1)
        self.assertTrue(res2)
        self.assertFalse(res3)
        self.assertFalse(res4)

    def test_isgoal_bw_numeric_margins(self):
        # Arrange
        g = Goal({'v4': ContinuousSet(1, 3)})

        i = Gaussian(1.3, .5).sample(500)
        i_ = Numeric()
        i_ = i_.fit(i.reshape(-1, 1), col=0)

        v4 = Gaussian(1.3, .51).sample(500)
        v4_ = Numeric()
        v4_ = v4_.fit(v4.reshape(-1, 1), col=0)

        v5 = Gaussian(2.8, .5).sample(500)
        v5_ = Numeric()
        v5_ = v5_.fit(v5.reshape(-1, 1), col=0)

        i = State({'v4': i_})

        astar1 = SubAStarBW_(
            i,
            g,
            models=self.models
        )

        # Act
        res1 = astar1.isgoal(Node(state=State({'v4': v4_}), g=0, h=0))
        res2 = astar1.isgoal(Node(state=State({'v4': v5_}), g=0, h=0))

        # Assert
        self.assertTrue(res1)
        self.assertFalse(res2)

    def test_isgoal_bw_multinomial(self):
        # Arrange
        i = SymbolicType('MultiVar', labels=["green", "red", "blue", "yellow", "black", "orange"])().set([.4, .2, .1, .1, .1, .1])
        v2_ = SymbolicType('MultiVar', labels=["green", "red", "blue", "yellow", "black", "orange"])().set([.3, .3, .1, .1, .1, .1])
        v3_ = SymbolicType('MultiVar', labels=["green", "red", "blue", "yellow", "black", "orange"])().set([.2, .1, .1, .3, .1, .2])

        s1 = State({'v2': v2_})
        s2 = State({'v2': v3_})

        astar1 = SubAStarBW_(
            State({'v2': i}),
            Goal({'v2': {"red", "green", "blue"}}),
            models=self.models
        )

        # Act
        res1 = astar1.isgoal(Node(state=s1, g=0, h=0))
        res2 = astar1.isgoal(Node(state=s2, g=0, h=0))

        # Assert
        self.assertTrue(res1)
        self.assertFalse(res2)

    def test_isgoal_bw_multiple(self):
        # Arrange
        g = Goal()
        g.update({
            'v1': {True},
            'v2': {"green", "red", "blue"},
            'v3': {1, 2, 5},
            'v4': ContinuousSet(1, 3)
        })

        v1_ = SymbolicVariable('BoolVar', Bool)
        v1_ = v1_.distribution().set(2/3)

        v2_ = SymbolicType('MultiVar', labels=["green", "red", "blue", "yellow", "black", "orange"])().set([.3, .3, .1, .1, .1, .1])
        v3_ = SymbolicType('IntVar', labels=[1, 2, 3, 4, 5])().set([.2, .2, .1, .1, .4])

        v4 = Gaussian(1.3, .4).sample(500)
        v4_ = Numeric()
        v4_ = v4_.fit(v4.reshape(-1, 1), col=0)

        s1 = State()
        s1.update({
            'v1': v1_,
            'v2': v2_,
            'v3': v3_,
            'v4': v4_
        })

        astar1 = SubAStarBW_(
            s1,
            g,
            models=self.models
        )

        # Act
        res1 = astar1.isgoal(Node(state=s1, g=0, h=0))

        # Assert
        self.assertTrue(res1)

    @unittest.skip
    def test_astar_single_action_update(self) -> None:
        s0 = self.init  # = [-61, 61, 0, -1]
        tm = self.models['move']
        tt = self.models['turn']

        # ACTION I: MOVE ==============================================================
        # plot init position distribution
        d = pd.DataFrame(
            data=[gendata('x_in', 'y_in', s0)],
            columns=['x', 'y', 'z', 'lbl']
        )
        plot_heatmap(
            xvar='x',
            yvar='y',
            data=d,
            title="$\\text{Init Distribution } P(x_{in},y_{in})$",
            limx=(0, 100),
            limy=(0, 100),
            show=True
        )

        # get update dist
        evidence = {
            var: ContinuousSet(s0[var].ppf(.05), s0[var].ppf(.95)) for var in s0.keys()
        }
        tm_ = tm.conditional_jpt(
            evidence=tm.bind(
                {k: v for k, v in evidence.items() if k in tm.varnames},
                allow_singular_values=False
            ),
            fail_on_unsatisfiability=False
        )
        dist_u1 = tm_.posterior(variables=tm.targets)

        # plot position update distribution
        d = pd.DataFrame(
            data=[gendata('x_out', 'y_out', dist_u1)],
            columns=['x', 'y', 'z', 'lbl'])
        plot_heatmap(
            xvar='x',
            yvar='y',
            data=d,
            title="$\delta\\text{-Distribution } P(x_{out},y_{out})$",
            limx=(-3, 3),
            limy=(-3, 3),
            show=True
        )

        # update pos
        # create successor state
        s1 = State()
        s1.update({k: v for k, v in s0.items()})

        n_segments = 20
        dist_u1['x_out'] = dist_u1['x_out'].approximate(n_segments=n_segments)
        s1['x_in'] = s1['x_in'] + dist_u1['x_out']
        s1['x_in'] = s1['x_in'].approximate(n_segments=n_segments)

        dist_u1['y_out'] = dist_u1['y_out'].approximate(n_segments=n_segments)
        s1['y_in'] = s1['y_in'] + dist_u1['y_out']
        s1['y_in'] = s1['y_in'].approximate(n_segments=n_segments)

        # plot result
        d = pd.DataFrame(
            data=[gendata('x_in', 'y_in', s1)],
            columns=['x', 'y', 'z', 'lbl'])
        plot_heatmap(
            xvar='x',
            yvar='y',
            data=d,
            title="$\\text{Init} + \delta$",
            limx=(0, 100),
            limy=(0, 100),
            show=True
        )

        # ACTION II: TURN ==============================================================
        # plot init position distribution
        d = pd.DataFrame(
            data=[gendata('xdir_in', 'ydir_in', s0)],
            columns=['x', 'y', 'z', 'lbl']
        )
        plot_heatmap(
            xvar='x',
            yvar='y',
            data=d,
            title="$\\text{Init Distribution } P(xdir_{in},ydir_{in})$",
            limx=(-3, 3),
            limy=(-3, 3),
            show=True
        )

        evidence = {
            var: ContinuousSet(s1[var].ppf(.05), s1[var].ppf(.95)) for var in s1.keys()
        }
        evidence['angle'] = -20
        tt_ = tt.conditional_jpt(
            evidence=tt.bind(
                {k: v for k, v in evidence.items() if k in tt.varnames},
                allow_singular_values=False
            ),
            fail_on_unsatisfiability=False
        )
        dist_u2 = tt_.posterior(variables=tt.targets)

        # plot direction update distribution
        d = pd.DataFrame(
            data=[gendata('xdir_out', 'ydir_out', dist_u2)],
            columns=['x', 'y', 'z', 'lbl'])
        plot_heatmap(
            xvar='x',
            yvar='y',
            data=d,
            title="$\delta\\text{-Distribution } P(xdir_{out},ydir_{out})$",
            limx=(-3, 3),
            limy=(-3, 3),
            show=True
        )

        # update dir
        # create successor state
        s2 = State()
        s2.update({k: v for k, v in s1.items()})

        n_segments = 20
        dist_u2['xdir_out'] = dist_u2['xdir_out'].approximate(n_segments=n_segments)
        s2['xdir_in'] = s2['xdir_in'] + dist_u2['xdir_out']
        s2['xdir_in'] = s2['xdir_in'].approximate(n_segments=n_segments)

        dist_u2['ydir_out'] = dist_u2['ydir_out'].approximate(n_segments=n_segments)
        s2['ydir_in'] = s2['ydir_in'] + dist_u2['ydir_out']
        s2['ydir_in'] = s2['ydir_in'].approximate(n_segments=n_segments)

        # plot result
        d = pd.DataFrame(
            data=[gendata('xdir_in', 'ydir_in', s2)],
            columns=['x', 'y', 'z', 'lbl'])
        plot_heatmap(
            xvar='x',
            yvar='y',
            data=d,
            title="$\\text{Init} + \delta$",
            limx=(-3, 3),
            limy=(-3, 3),
            show=True
        )

    # @unittest.skip
    def test_astar_single_predecessor_update(self) -> None:
        initx, inity, initdirx, initdiry = [3.5, 58.5, .75, .75]  # [3, 60, 1, 0]
        goalx, goaly = [6, 60]
        tolerance_pos = 0.05
        tolerance_dir = .01
        tol = .5

        dx = Gaussian(initx, tolerance_pos).sample(500)
        distx = Numeric()
        distx.fit(dx.reshape(-1, 1), col=0)

        dy = Gaussian(inity, tolerance_pos).sample(500)
        disty = Numeric()
        disty.fit(dy.reshape(-1, 1), col=0)

        ddx = Gaussian(initdirx, tolerance_dir).sample(500)
        distdx = Numeric()
        distdx.fit(ddx.reshape(-1, 1), col=0)

        ddy = Gaussian(initdiry, tolerance_dir).sample(500)
        distdy = Numeric()
        distdy.fit(ddy.reshape(-1, 1), col=0)

        self.a_star = SubAStarBW_(
            State(
                {
                    'x_in': distx,
                    'y_in': disty,
                    'xdir_in': distdx,
                    'ydir_in': distdy
                }
            ),
            Goal(
                {
                    'x_in': ContinuousSet(goalx - tol, goalx + tol),
                    'y_in': ContinuousSet(goaly - tol, goaly + tol)
                }
            ),
            # Goal(
            #     {
            #         'x_in': ContinuousSet(4.5, 5.5),
            #         'y_in': ContinuousSet(59.5, 60.5)
            #     }
            # ),
            models=self.models
        )
        self.path = list(self.a_star.search())

        # # def dist(s1, s2):
        # #     # return mean of distances to each required variable in init state (=goal in reverse)
        # #     dists = []
        # #     for k, v in s2.items():
        # #         if k in s1:
        # #             dists.append(Numeric.distance(s1[k], v))
        # #     return np.mean(dists)
        #
        # num_preds = 50
        #
        # query = {
        #     var:
        #         self.goal[var] if isinstance(self.goal[var], (set, ContinuousSet)) else
        #         self.goal[var].mpe()[0] for var in self.goal.keys()
        # }
        #
        # # predecessor from move action model
        # q_ = tm.bind(
        #     {
        #         k: v for k, v in query.items() if k in tm.varnames
        #     },
        #     allow_singular_values=False
        # )
        #
        # # Transform into internal values/intervals (symbolic values to their indices)
        # query_ = tm._preprocess_query(
        #     q_,
        #     skip_unknown_variables=True
        # )
        #
        # for i, var in enumerate(tm.variables):
        #     if var in query_: continue
        #     if var.numeric:
        #         query_[var] = R
        #     else:
        #         query_[var] = set(var.domain.labels.values())
        #
        # def determine_leaf_confs(l):
        #     #  assuming, that the _out variables are deltas and querying for _in semantically means querying for the
        #     #  result of adding the delta to the _in variable (i.e. the actual outcome of performing the action
        #     #  represented by the leaf)
        #     conf = defaultdict(float)
        #     s_ = State()
        #     s_.tree = "move"
        #     s_.leaf = l.idx
        #
        #     if l.idx in [3476, 3791, 4793]:
        #         pass
        #     for v, _ in l.distributions.items():
        #         vname = v.name
        #         invar = vname.replace('_out', '_in')
        #         outvar = vname.replace('_in', '_out')
        #
        #         if vname.endswith('_in') and vname.replace('_in', '_out') in l.distributions:
        #             # if the current variable is an _in variable, and contains the respective _out variable
        #             # distribution, add the two distributions and calculate probability on resulting
        #             # distribution
        #             outdist = l.distributions[outvar]
        #             indist = l.distributions[invar]
        #
        #             # determine probability that this action (leaf) produces desired output for this variable
        #             tmp_dist = indist + outdist
        #             c_ = tmp_dist.p(query_[vname])
        #
        #             # determine distribution from which the execution of this action (leaf) produces desired output
        #             try:
        #                 cond = tmp_dist.crop(query_[vname])
        #                 tmp_diff = cond - outdist
        #                 tmp_diff = tmp_dist.approximate(n_segments=20)
        #                 d_ = tmp_diff
        #             except Unsatisfiability:
        #                 c_ = 0
        #         else:
        #             # default case
        #             c_ = l.distributions[vname].p(query_[vname])
        #             d_ = l.distributions[vname]
        #
        #         # drop entire leaf if only one variable has probability 0
        #         if not c_ > 0:
        #             return
        #         conf[vname] = c_
        #         s_[vname] = d_
        #     return s_, conf
        #
        # # find the leaf (or the leaves) that matches the query best
        # steps = []
        # for i, (k, l) in enumerate(tm.leaves.items()):
        #     res = determine_leaf_confs(l)
        #     if res is not None:
        #         steps.append(res)
        #
        # # sort candidates according to overall confidence (=probability to reach) and select `num_preds` best ones
        # selected_steps = sorted(steps, reverse=True, key=lambda x: np.product([v for _, v in x[1].items()]))[:num_preds]
        #
        # # add info so selected_steps contains tuples (step, confidence, distance to init state, leaf prior)
        # selected_steps_ = [(s, c, dist(s, self.init), tm.leaves[s.leaf].prior) for s, c in selected_steps]
        #
        # # sort remaining candidates according to distance to init state (=goal in reverse)
        # selected_steps_wasserstein = sorted(selected_steps_, key=lambda x: x[2])  # zum start belief state
        #
        # # TODO: for the first num_preds steps, plot leaf distributions for xin/yin variables
        #
        # d = pd.DataFrame(
        #     data=[
        #         gendata(
        #             'x_in',
        #             'y_in',
        #             state=state,
        #             params={
        #                 "confidence": "<br>".join([f"{k}: {v}" for k,v in conf.items()]),
        #                 "leaf prior": prior,
        #                 "distance": dist,
        #                 "MPE x_in": state["x_in"].mpe()[0],
        #                 "MPE y_in": state["y_in"].mpe()[0],
        #                 "EXP x_in": state["x_in"].expectation(),
        #                 "EXP y_in": state["y_in"].expectation(),
        #             }
        #         ) for (state, conf, dist, prior) in selected_steps_wasserstein
        #     ],
        #     columns=['x', 'y', 'z', 'lbl'])
        #
        # plot_heatmap(
        #     xvar='x',
        #     yvar='y',
        #     data=d,
        #     title="$\\text{Predecessors } P(x_{in},y_{in})$",
        #     limx=(0, 100),
        #     limy=(0, 100),
        #     save=os.path.join(locs.logs, f"test_astar_single_predecessor_update-absolute.html"),
        #     show=True,
        #     fun="heatmap"
        # )
        #
        # d = pd.DataFrame(
        #     data=[
        #         gendata(
        #             'x_out',
        #             'y_out',
        #             state=state,
        #             params={
        #                 "confidence": "<br>".join([f"{k}: {v}" for k, v in conf.items()]),
        #                 "leaf prior": prior,
        #                 "distance": dist,
        #                 "MPE x_out": state["x_out"].mpe()[0],
        #                 "MPE y_out": state["y_out"].mpe()[0],
        #                 "EXP x_out": state["x_out"].expectation(),
        #                 "EXP y_out": state["y_out"].expectation(),
        #             }
        #         ) for (state, conf, dist, prior) in selected_steps_wasserstein
        #     ],
        #     columns=['x', 'y', 'z', 'lbl'])
        #
        # plot_heatmap(
        #     xvar='x',
        #     yvar='y',
        #     data=d,
        #     title="$\\text{Predecessors } P(x_{out},y_{out})$",
        #     limx=(-3, 3),
        #     limy=(-3, 3),
        #     save=os.path.join(locs.logs, f"test_astar_single_predecessor_update-delta.html"),
        #     show=True,
        #     fun="heatmap"
        # )

    def tearDown(self) -> None:
        # draw path steps into grid (use action symbols)
        print()

    @classmethod
    def tearDownClass(cls) -> None:
        print()
