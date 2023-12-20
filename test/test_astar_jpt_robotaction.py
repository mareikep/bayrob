import os
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from jpt.base.intervals import ContinuousSet, R

from calo.application.astar_jpt_app import State_, SubAStar_, SubAStarBW_
from calo.core.astar import BiDirAStar, Node
from calo.core.astar_jpt import Goal, State
from calo.utils import locs
from calo.utils.plotlib import gendata, plot_heatmap
from calo.utils.utils import recent_example
from jpt import JPT
from jpt.base.errors import Unsatisfiability
from jpt.distributions import Numeric, Gaussian


class AStarRobotActionJPTTests(unittest.TestCase):

    # assumes trees robotaction_move and robotaction_move generated previously from functions in calo-dev/test/test_robotpos.py,
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

        cls.models = dict(
            [
                (
                    Path(p).parent.name,
                    JPT.load(str(treefile))
                )
                for p in [cls.recent_move, cls.recent_turn, cls.recent_perc]
                for treefile in Path(p).rglob('*.tree')
            ]
        )

        cls.init = State_()
        cls.goal = Goal()

        # plot initial distributions over x/y positions
        t = cls.models['move']
        # plot_tree_dist(
        #     tree=t,
        #     qvarx=t.varnames['x_in'],
        #     qvary=t.varnames['y_in'],
        #     title='Initial distribution P(x,y)',
        #     limx=(0, 100),
        #     limy=(0, 100),
        #     save=os.path.join(os.path.join(recent, 'plots', '000-init-dist.html')),
        #     show=True
        # )

        initx, inity, initdirx, initdiry = [3, 60, 1, 0]
        goalx, goaly = [5, 60]
        tolerance_pos = 2
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


        cls.init.update(
            {
                'x_in': distx,
                'y_in': disty,
                'xdir_in': distdx,
                'ydir_in': distdy
            }
        )
        # cls.initstate.plot(show=True)

        cls.goal.update(
            {
                'x_in': ContinuousSet(goalx - tol, goalx + tol),
                'y_in': ContinuousSet(goaly - tol, goaly + tol)
            }
        )

    def test_isgoal_fw_t(self) -> None:
        self.a_star = SubAStar_(
            self.init,
            self.goal,
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
    def test_astar_fw_path(self) -> None:
        self.a_star = SubAStar_(
            self.init,
            self.goal,
            models=self.models
        )
        self.path = list(self.a_star.search())

    # @unittest.skip
    def test_astar_bw_path(self) -> None:
        self.a_star = SubAStarBW_(
            self.init,
            self.goal,
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

    # @unittest.skip
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
        s1 = State_()
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
        s2 = State_()
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

    def test_astar_single_predecessor_update(self) -> None:
        tm = self.models['move']
        print(self.goal)

        def dist(s1, s2):
            # return mean of distances to each required variable in init state (=goal in reverse)
            dists = []
            for k, v in s2.items():
                if k in s1:
                    dists.append(Numeric.distance(s1[k], v))
            return np.mean(dists)

        num_preds = 50

        query = {
            var:
                self.goal[var] if isinstance(self.goal[var], (set, ContinuousSet)) else
                self.goal[var].mpe()[0] for var in self.goal.keys()
        }

        # predecessor from move action model
        q_ = tm.bind(
            {
                k: v for k, v in query.items() if k in tm.varnames
            },
            allow_singular_values=False
        )

        # Transform into internal values/intervals (symbolic values to their indices)
        query_ = tm._preprocess_query(
            q_,
            skip_unknown_variables=True
        )

        for i, var in enumerate(tm.variables):
            if var in query_: continue
            if var.numeric:
                query_[var] = R
            else:
                query_[var] = set(var.domain.labels.values())

        def determine_leaf_confs(l):
            #  assuming, that the _out variables are deltas and querying for _in semantically means querying for the
            #  result of adding the delta to the _in variable (i.e. the actual outcome of performing the action
            #  represented by the leaf)
            conf = defaultdict(float)
            s_ = State()
            s_.tree = "move"
            s_.leaf = l.idx

            if l.idx in [3476, 3791, 4793]:
                pass
            for v, _ in l.distributions.items():
                vname = v.name
                invar = vname.replace('_out', '_in')
                outvar = vname.replace('_in', '_out')

                if vname.endswith('_in') and vname.replace('_in', '_out') in l.distributions:
                    # if the current variable is an _in variable, and contains the respective _out variable
                    # distribution, add the two distributions and calculate probability on resulting
                    # distribution
                    outdist = l.distributions[outvar]
                    indist = l.distributions[invar]

                    # determine probability that this action (leaf) produces desired output for this variable
                    tmp_dist = indist + outdist
                    c_ = tmp_dist.p(query_[vname])

                    # determine distribution from which the execution of this action (leaf) produces desired output
                    try:
                        cond = tmp_dist.crop(query_[vname])
                        tmp_diff = cond - outdist
                        tmp_diff = tmp_dist.approximate(n_segments=20)
                        d_ = tmp_diff
                    except Unsatisfiability:
                        c_ = 0
                else:
                    # default case
                    c_ = l.distributions[vname].p(query_[vname])
                    d_ = l.distributions[vname]

                # drop entire leaf if only one variable has probability 0
                if not c_ > 0:
                    return
                conf[vname] = c_
                s_[vname] = d_
            return s_, conf

        # find the leaf (or the leaves) that matches the query best
        steps = []
        for i, (k, l) in enumerate(tm.leaves.items()):
            res = determine_leaf_confs(l)
            if res is not None:
                steps.append(res)

        # sort candidates according to overall confidence (=probability to reach) and select `num_preds` best ones
        selected_steps = sorted(steps, reverse=True, key=lambda x: np.product([v for _, v in x[1].items()]))[:num_preds]

        # add info so selected_steps contains tuples (step, confidence, distance to init state, leaf prior)
        selected_steps_ = [(s, c, dist(s, self.init), tm.leaves[s.leaf].prior) for s, c in selected_steps]

        # sort remaining candidates according to distance to init state (=goal in reverse)
        selected_steps_wasserstein = sorted(selected_steps_, key=lambda x: x[2])  # zum start belief state

        # TODO: for the first num_preds steps, plot leaf distributions for xin/yin variables

        d = pd.DataFrame(
            data=[
                gendata(
                    'x_in',
                    'y_in',
                    state=state,
                    params={
                        "confidence": "<br>".join([f"{k}: {v}" for k,v in conf.items()]),
                        "leaf prior": prior,
                        "distance": dist,
                        "MPE x_in": state["x_in"].mpe()[0],
                        "MPE y_in": state["y_in"].mpe()[0],
                        "EXP x_in": state["x_in"].expectation(),
                        "EXP y_in": state["y_in"].expectation(),
                    }
                ) for (state, conf, dist, prior) in selected_steps_wasserstein
            ],
            columns=['x', 'y', 'z', 'lbl'])

        plot_heatmap(
            xvar='x',
            yvar='y',
            data=d,
            title="$\\text{Predecessors } P(x_{in},y_{in})$",
            limx=(0, 100),
            limy=(0, 100),
            save=os.path.join(locs.logs, f"test_astar_single_predecessor_update-absolute.html"),
            show=True,
            fun="heatmap"
        )

        d = pd.DataFrame(
            data=[
                gendata(
                    'x_out',
                    'y_out',
                    state=state,
                    params={
                        "confidence": "<br>".join([f"{k}: {v}" for k, v in conf.items()]),
                        "leaf prior": prior,
                        "distance": dist,
                        "MPE x_out": state["x_out"].mpe()[0],
                        "MPE y_out": state["y_out"].mpe()[0],
                        "EXP x_out": state["x_out"].expectation(),
                        "EXP y_out": state["y_out"].expectation(),
                    }
                ) for (state, conf, dist, prior) in selected_steps_wasserstein
            ],
            columns=['x', 'y', 'z', 'lbl'])

        plot_heatmap(
            xvar='x',
            yvar='y',
            data=d,
            title="$\\text{Predecessors } P(x_{out},y_{out})$",
            limx=(-3, 3),
            limy=(-3, 3),
            save=os.path.join(locs.logs, f"test_astar_single_predecessor_update-delta.html"),
            show=True,
            fun="heatmap"
        )

    def tearDown(self) -> None:
        # draw path steps into grid (use action symbols)
        print()

    @classmethod
    def tearDownClass(cls) -> None:
        print()
