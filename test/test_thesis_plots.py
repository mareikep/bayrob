import math
import os
import unittest

import unittest
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from pandas import DataFrame

from calo.application.astar_jpt_app import State_
from calo.utils import locs
from calo.utils.plotlib import plot_heatmap, plot_data_subset, plot_tree_dist, plot_pos, plot_path, defaultconfig, \
    plotly_animation, plot_scatter_quiver, plot_dir
from calo.utils.utils import recent_example
from jpt import SymbolicType, NumericVariable, JPT
from jpt.base.intervals import ContinuousSet, RealSet
from jpt.distributions import Gaussian, Numeric
from jpt.distributions.quantile.quantiles import QuantileDistribution


def generate_gaussian_samples(gaussians, n):
    per_gaussian = int(n / len(gaussians))
    data = [g.sample(per_gaussian) for g in gaussians]
    colors = [[c] * per_gaussian for c in ['#ff0000', '#0000ff', '#00ff00', '#0f0f0f', '#f0f0f0'][:len(gaussians)]]

    all_data = np.vstack(data)
    for d, c in zip(data, colors):
        plt.scatter(d[:, 0], d[:, 1], color=c, marker='x')
    # plt.scatter(gauss2_data[:, 0], gauss2_data[:, 1], color='b', marker='x')
    # all_data = np.hstack([all_data, reduce(list.__add__, colors)])

    df = DataFrame({'X': all_data[:, 0], 'Y': all_data[:, 1], 'Color': reduce(list.__add__, colors)})
    return df

class ThesisPlotsTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
       cls.defaultconfig = dict(
            displaylogo=False,
            toImageButtonOptions=dict(
                format='svg',  # one of png, svg, jpeg, webp
                filename='calo_plot',
                height=500,
                width=700,
                scale=1  # Multiply title/legend/axis/canvas sizes by this factor
            )
       )

       cls.recent_move = recent_example(os.path.join(locs.examples, 'robotaction_move'))
       cls.recent_turn = recent_example(os.path.join(locs.examples, 'robotaction_turn'))
       # recent = os.path.join(locs.examples, 'robotaction_move', '2023-11-18_15:24')
       print("loading examples from", cls.recent_move, cls.recent_turn)

       cls.models = dict(
           [
               (
                   treefile.name,
                   JPT.load(str(treefile))
               )
               for p in [cls.recent_move, cls.recent_turn]
               for treefile in Path(p).rglob('*.tree')
           ]
       )

    def test_plot_init_dist(self) -> None:
        t = ThesisPlotsTests.models['000-robotaction_move.tree']
        plot_tree_dist(
            tree=t,
            qvarx=t.varnames['x_in'],
            qvary=t.varnames['y_in'],
            title='Initial distribution P(x,y)',
            limx=(-100, 100),
            limy=(-100, 100),
            save=os.path.join(os.path.join(locs.logs, 'init-dist.html')),
            show=True
        )

    def test_plot_dist_similarity_discrete(self) -> None:
        # Arrange
        DistA = SymbolicType('DistA', labels=['A', 'B', 'C'])
        DistB = SymbolicType('DistB', labels=['D', 'E', 'F'])
        DistC = SymbolicType('DistC', labels=['C', 'D', 'E'])
        d1 = DistA().set(params=[.5, .25, .25])
        d2 = DistB().set(params=[.5, .25, .25])
        d3 = DistA().set(params=[.2, .1, .7])
        d4 = DistC().set(params=[.25, .1, .65])

        from jpt.distributions import Multinomial
        d12 = Multinomial.jaccard_similarity(d1, d2)
        d13 = Multinomial.jaccard_similarity(d1, d3)
        d14 = Multinomial.jaccard_similarity(d1, d4)
        d23 = Multinomial.jaccard_similarity(d2, d3)
        d24 = Multinomial.jaccard_similarity(d2, d4)
        d34 = Multinomial.jaccard_similarity(d3, d4)


        print(d12, d13, d14, d23, d24, d34)

        # Act
        mainfig = go.Figure()

        d1fig = d1.plot(
            title="$\\text{dist}_1$",
            color="rgb(0,104,180)",
            view=False,
            horizontal=False
        )

        d2fig = d2.plot(
            title="$\\text{dist}_2$",
            color="rgb(134, 129, 177)",
            view=False,
            horizontal=False
        )

        d3fig = d3.plot(
            title="$\\text{dist}_3$",
            color="rgb(138, 203, 183)",
            view=False,
            horizontal=False
        )

        mainfig.add_traces(
            data=d1fig.data
        )

        mainfig.add_traces(
            data=d2fig.data
        )

        mainfig.add_traces(
            data=d3fig.data
        )

        mainfig.update_layout(
            xaxis=dict(
                title='$\\text{label}$',
                range=None, #  ['A', 'B', 'C', 'D', 'E', 'F'],
                ticks="outside",
                tickson="boundaries",
                ticklen=20
            ),
            yaxis=dict(
                title='$P(\\text{label})$',
                range=[0, 1]
            ),
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.01
            )
        )


        mainfig.write_image(
            os.path.join(locs.logs, f'similarity_discrete.svg'),
            scale=1
        )

        mainfig.write_html(
            os.path.join(locs.logs, f'similarity_discrete.html'),
            config=ThesisPlotsTests.defaultconfig,
            include_plotlyjs="cdn"
        )

        mainfig.show(
            config=ThesisPlotsTests.defaultconfig
        )

        # Assert

    def test_plot_dist_add_continuous(self) -> None:
        # Arrange

        mu1, mu2 = [-2, 1]
        v1, v2 = [2, .02]

        dx = Gaussian(mu1, v1).sample(500)
        distx = Numeric()
        distx.fit(dx.reshape(-1, 1), col=0)

        dxdelta = Gaussian(mu2, v2).sample(500)
        distxdelta = Numeric()
        distxdelta.fit(dxdelta.reshape(-1, 1), col=0)

        distx_ = distx + distxdelta

        # Act
        mainfig = go.Figure()

        d1fig = distx.plot(
            title=f"$X \sim \cal{{N}}({mu1},{v1})$",
            color="0,104,180",
            view=False,
        )

        mainfig.add_traces(
            data=d1fig.data
        )

        d2fig = distx_.plot(
            title=f"$X' = X_ + X_{{delta}} \sim \cal{{N}}({mu2},{v2})$",
            color="134, 129, 177",
            view=False,
        )

        mainfig.add_traces(
            data=d2fig.data
        )

        mainfig.update_layout(
            xaxis=dict(
                title='$x$',
                range=[-7, 4]
            ),
            yaxis=dict(
                title='$P(x)$',
                range=[0, 1]
            ),
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.01
            )
        )

        mainfig.write_image(
            os.path.join(locs.logs, f'addition_continuous.svg'),
            scale=1
        )

        mainfig.write_html(
            os.path.join(locs.logs, f'addition_continuous.html'),
            config=ThesisPlotsTests.defaultconfig,
            include_plotlyjs="cdn"
        )

        mainfig.show(
            config=ThesisPlotsTests.defaultconfig
        )

    def test_plot_dist_similarity_continuous(self) -> None:
        # Arrange

        mu1, mu2 = [-2, 0.5]
        v1, v2 = [2, .4]
        gauss1 = Gaussian(mu1, v1)
        gauss2 = Gaussian(mu2, v2)
        gauss3 = Gaussian(mu1 + mu2, v1 + v2)

        x = np.linspace(-9, 9, 300)
        pdfg1 = gauss1.pdf(x)
        pdfg2 = gauss2.pdf(x)
        sumg1g2 = gauss3.pdf(x)
        mixture = [gauss1.pdf(d) + gauss2.pdf(d) for d in x]  # mixture

        # Act
        mainfig = go.Figure()

        mainfig.add_trace(
            go.Scatter(
                x=x,
                y=pdfg1,
                mode='lines',
                name=f'$X_1 \sim \cal{{N}}({mu1},{v1})$',
                line=dict(
                    color='rgba(0,104,180,1)',
                    width=4,
                )
            )
        )

        mainfig.add_trace(
            go.Scatter(
                x=x,
                y=pdfg2,
                mode='lines',
                name=f'$X_2 \sim \cal{{N}}({mu2},{v2})$',
                line=dict(
                    color='rgba(134, 129, 177,1)',
                    width=4,
                )
            )
        )

        mainfig.add_trace(
            go.Scatter(
                x=x,
                y=mixture,
                mode='lines',
                name='$\\text{Mixture of} X_1 \\text{and} X_2$',
                line=dict(
                    color='rgba(138, 203, 183,1)',
                    width=4,
                    dash="dot"
                )
            )
        )

        mainfig.add_trace(
            go.Scatter(
                x=x,
                y=sumg1g2,
                mode='lines',
                name=f'$Z = X_1 + X_2 \sim \cal{{N}}({mu1+mu2},{v1+v2})$',
                line=dict(
                    color='rgba(244, 161, 152,1)',
                    width=4,
                    dash="dot"
                )
            )
        )

        # mainfig.add_trace(
        #     go.Scatter(
        #         x=x,
        #         y=cdfg1,
        #         mode='lines',
        #         name='CDF of G1',
        #         line=dict(
        #             color='rgba(0,104,180,1)',
        #             width=4,
        #             dash="dot"
        #         )
        #     )
        # )
        #
        # mainfig.add_trace(
        #     go.Scatter(
        #         x=x,
        #         y=cdfg2,
        #         mode='lines',
        #         name='CDF of G2',
        #         line=dict(
        #             color='rgba(134, 129, 177,1)',
        #             width=4,
        #             dash="dot"
        #         )
        #     )
        # )

        mainfig.update_layout(
            xaxis=dict(
                title='$x$',
                range=[-7, 4]
            ),
            yaxis=dict(
                title='$P(x)$',
                range=[0, 1]
            ),
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.01
            )
        )

        mainfig.write_image(
            os.path.join(locs.logs, f'similarity_continuous.svg'),
            scale=1
        )

        mainfig.write_html(
            os.path.join(locs.logs, f'similarity_continuous.html'),
            config=ThesisPlotsTests.defaultconfig,
            include_plotlyjs="cdn"
        )

        mainfig.show(
            config=ThesisPlotsTests.defaultconfig
        )


    def test_reproduce_data_find_limits(self) -> None:
        # -> constrain target pos, plot feature pos

        # load data and JPT that has been learnt from this data
        j = ThesisPlotsTests.models['000-robotaction_move.tree']
        print(f"Loading tree from {ThesisPlotsTests.recent}")

        # set settings
        limx = (0, 100)
        limy = (0, 100)

        x_out = 0
        y_out = 0

        # leave untouched
        tolerance = .05

        xoutmin = x_out - tolerance
        xoutmax = x_out + tolerance
        youtmin = y_out - tolerance
        youtmax = y_out + tolerance

        pdfvars = {
            'x_out': 0,#ContinuousSet(xoutmin, xoutmax),
            'y_out': 0,#ContinuousSet(youtmin, youtmax),
            'collided': True
        }

        # constraints is a list of 3-tuples: ('<column name>', 'operator', value)
        # constraints = [(var, op, v) for var, val in pdfvars.items() for v, op in [(val.lower, ">="), (val.upper, "<=")]]
        # print('\nConstraints on dataset: ', constraints)

        # generate tree conditioned on given position and/or direction
        cond = j.conditional_jpt(
            evidence=j.bind(
                {k: v for k, v in pdfvars.items() if k in j.varnames},
                allow_singular_values=False
            ),
            fail_on_unsatisfiability=False
        )

        # cond.plot(plotvars=cond.variables, view=True)

        # data generation
        x = np.linspace(*limx, 200)
        y = np.linspace(*limy, 200)

        X, Y = np.meshgrid(x, y)
        Z = np.array(
            [
                cond.pdf(
                    cond.bind(
                        {
                            'x_in': x,
                            'y_in': y
                        }
                    )
                ) for x, y, in zip(X.ravel(), Y.ravel())
            ]
        ).reshape(X.shape)
        lbl = np.full(Z.shape, '<br>'.join([f'{vname}: {val}' for vname, val in pdfvars.items()]))

        data = pd.DataFrame(
            data=[[x, y, Z, lbl]],
            columns=['x', 'y', 'z', 'lbl']
        )

        prefix = f'{"_".join([f"{k}({v})" for k,v in pdfvars.items()])}'

        # plot JPT
        plot_heatmap(
            xvar='x',
            yvar='y',
            data=data,
            limx=limx,
            limy=limy,
            limz=(0, 0.0002),
            save=os.path.join(locs.logs, f"boundaries-{prefix}-dist-hm.html"),
            show=True,
        )

        plot_heatmap(
            xvar='x',
            yvar='y',
            data=data,
            limx=limx,
            limy=limy,
            limz=(0, 0.0002),
            save=os.path.join(locs.logs, f"boundaries-{prefix}-dist-surface.html"),
            show=True,
            fun="surface"
        )

        # fig.write_html(
        #     os.path.join(locs.logs, f'crampath.html'),
        #     config=defaultconfig,
        #     include_plotlyjs="cdn"
        # )

        # plot ground truth
        df = pd.read_parquet(os.path.join(ThesisPlotsTests.recent, 'data', f'000-robotaction_move.parquet'))
        plot_data_subset(
            df,
            xvar="x_in",
            yvar="y_in",
            constraints=pdfvars,
            limx=limx,
            limy=limy,
            save=os.path.join(locs.logs, f"boundaries-{prefix}-gt.html"),
            show=True
        )

    def test_reproduce_data_single_jpt(self) -> None:
        # -> constrain feature pos, plot target pos

        # load data and JPT that has been learnt from this data
        j = ThesisPlotsTests.models['000-robotaction_move.tree']
        print(f"Loading tree from {ThesisPlotsTests.recent}")

        # set settings
        limx = (-3, 3)
        limy = (-3, 3)

        x_in = -50
        y_in = 0
        xdir_in = -1
        ydir_in = 0
        x_out = 0
        y_out = 0

        # leave untouched
        tolerance = .05
        tolerance_ = 1

        xmin = x_in - tolerance_
        xmax = x_in + tolerance_
        ymin = y_in - tolerance_
        ymax = y_in + tolerance_

        xdirmin = xdir_in - tolerance
        xdirmax = xdir_in + tolerance
        ydirmin = ydir_in - tolerance
        ydirmax = ydir_in + tolerance

        xoutmin = x_out - tolerance_
        xoutmax = x_out + tolerance_
        youtmin = y_out - tolerance_
        youtmax = y_out + tolerance_

        pdfvars = {
            'x_in': ContinuousSet(xmin, xmax),
            # 'y_in': ContinuousSet(ymin, ymax),
            # 'xdir_in': ContinuousSet(xdirmin, xdirmax),
            # 'ydir_in': ContinuousSet(ydirmin, ydirmax),
            # 'x_out': ContinuousSet(xoutmin, xoutmax),
            # 'y_out': ContinuousSet(youtmin, youtmax),
            # 'collided': True
        }

        # generate tree conditioned on given position and/or direction
        cond = j.conditional_jpt(
            evidence=j.bind(
                {k: v for k, v in pdfvars.items() if k in j.varnames},
                allow_singular_values=False
            ),
            fail_on_unsatisfiability=False
        )

        # data generation
        x = np.linspace(*limx, 50)
        y = np.linspace(*limy, 50)

        X, Y = np.meshgrid(x, y)
        Z = np.array(
            [
                cond.pdf(
                    cond.bind(
                        {
                            'x_out': x,
                            'y_out': y
                        }
                    )
                ) for x, y, in zip(X.ravel(), Y.ravel())
            ]
        ).reshape(X.shape)
        lbl = np.full(Z.shape, '<br>'.join([f'{vname}: {val}' for vname, val in pdfvars.items()]))

        data = pd.DataFrame(
            data=[[x, y, Z, lbl]],
            columns=['x', 'y', 'z', 'lbl']
        )

        # plot JPT
        plot_heatmap(
            xvar='x',
            yvar='y',
            data=data,
            title=f'pdf({",".join([f"{vname}: {val}" for vname, val in pdfvars.items()])})',
            limx=limx,
            limy=limy,
            limz=(0, 0.0002),
            show=True,
        )

        # plot ground truth
        df = pd.read_parquet(os.path.join(ThesisPlotsTests.recent, 'data', f'000-robotaction_move.csv'))
        plot_data_subset(
            df,
            "x_out",
            "y_out",
            pdfvars,
            limx=limx,
            limy=limy,
            show=True
        )

    def test_face(self) -> None:
        j = ThesisPlotsTests.models['000-robotaction_move.tree']
        df = pd.read_parquet(os.path.join(ThesisPlotsTests.recent_move, 'data', f'000-robotaction_move.parquet'))
        tolerance = .3
        limx = (-3, 3)
        limy = (-3, 3)

        rx = RealSet([
            ContinuousSet(-.5 - tolerance, -.5 + tolerance),
            ContinuousSet(.5 - tolerance, .5 + tolerance)
        ])
        ry = RealSet([
            ContinuousSet(-1 - tolerance, -1 + tolerance),
            ContinuousSet(.5 - tolerance, .5 + tolerance),
            ContinuousSet(.5 - tolerance, .5 + tolerance)
        ])
        s = ('((`ydir_in` >= -1.3) & (`ydir_in` <= -0.7)) | '
             '((`xdir_in` >= -0.8) & (`xdir_in` <= 0.2) & (`ydir_in` >= 0.3) & (`ydir_in` <= 0.8)) | '
             '((`xdir_in` >= 0.3) & (`xdir_in` <= 0.8) & (`ydir_in` >= 0.3) & (`ydir_in` <= 0.8))')
        df = df.query(s)
        plot_data_subset(
            df,
            xvar='x_out',
            yvar='y_out',
            constraints={},
            limx=limx,
            limy=limy,
            show=True
        )

        # from jpt import infer_from_dataframe
        # variables = infer_from_dataframe(
        #     df,
        #     scale_numeric_types=False,
        #     # precision=.5
        # )
        #
        # jpt_ = JPT(
        #     variables=variables,
        #     targets=variables[4:],
        #     min_impurity_improvement=None,
        #     min_samples_leaf=400
        # )
        #
        # jpt_.learn(df, close_convex_gaps=False)
        #
        # jpt_.save(os.path.join(ThesisPlotsTests.recent_move, f'000-funnyface.tree'))
        #
        # jpt_.plot(
        #     title='funnyface',
        #     plotvars=list(jpt_.variables),
        #     filename=f'000-funnyface',
        #     directory=os.path.join(ThesisPlotsTests.recent_move, 'plots'),
        #     leaffill='#CCDAFF',
        #     nodefill='#768ABE',
        #     alphabet=True,
        #     view=False
        # )

        # generate tree conditioned on given position and/or direction
        jpt_ = j.conditional_jpt(
            evidence=j.bind(
                {
                    'xdir_in': rx,
                    'ydir_in': ry
                },
                allow_singular_values=False
            ),
            fail_on_unsatisfiability=False
        )

        # data generation
        x = np.linspace(*limx, 50)
        y = np.linspace(*limy, 50)

        X, Y = np.meshgrid(x, y)
        Z = np.array(
            [
                jpt_.pdf(
                    jpt_.bind(
                        {
                            'x_out': x,
                            'y_out': y
                        }
                    )
                ) for x, y, in zip(X.ravel(), Y.ravel())
            ]
        ).reshape(X.shape)
        lbl = np.full(Z.shape, ":)")

        data = pd.DataFrame(
            data=[[x, y, Z, lbl]],
            columns=['x', 'y', 'z', 'lbl']
        )

        plot_heatmap(
            xvar='x',
            yvar='y',
            data=data,
            title=None,  # f'pdf({",".join([f"{vname}: {val}" for vname, val in pdfvars.items()])})',
            limx=limx,
            limy=limy,
            show=True,
            showbuttons=True
        )

    def test_reproduce_data_multiple_jpt(self) -> None:
        # load data and JPT that has been learnt from this data
        j = ThesisPlotsTests.models['000-robotaction_move.tree']
        df = pd.read_parquet(os.path.join(ThesisPlotsTests.recent_move, 'data', f'000-robotaction_move.parquet'))

        # set settings
        limx = (-3, 3)
        limy = (-3, 3)

        o1 = [10, 10, 20, 20]  # chair1"
        o2 = [30, 10, 40, 20]  # "chair2"
        o3 = [10, 30, 50, 50]  # "kitchen_island"
        o4 = [80, 30, 100, 70]  # "stove"
        o5 = [10, 80, 50, 100]  # "kitchen_unit"
        o6 = [60, 80, 80, 100]  # "fridge"

        ox1, oy1, ox2, oy2 = o3

        # constraints/query values
        positions = {
            # "free-pos": [  # random position in obstacle-free area
                # (None, None, None, None),
                # (50,50, None, None),
                # (20, 70, -.7, -.7),
                # (20, 70, .7, -.7),
                # (20, 70, .7, .7),
                # (20, 70, -.7, .7),
            # ],
            "no-pos": [  # all directions without given pos
                (None, None, 0, -1),
                (None, None, 0, 1),
                (None, None, .5, -.5),
                (None, None, .5, .5),
                (None, None, -.5, -.5),
                (None, None, -.5, .5),
                (None, None, 1, 0),
                (None, None, -1, 0),
                (None, None, -1, None),
                (None, None, 1, None),
                (None, None, None, -1),
                (None, None, None, 1),
            ],
            # "grid-corners": [  # all corners of gridworld
            #     (0, 0, None, None),
            #     (0, 100, None, None),  # broken!
            #     (100, 0, None, None),  # broken!
            #     (100, 100, None, None)  # broken!
            # ],
            # "grid-edges": [  # all edges of gridworld (center)
            #     (0, 50, None, None),
            #     (100, 10, None, None),
            #     (50, 0, None, None),
            #     (50, 100, None, None)
            # ],
            # "obstacle-corners": [  # all corners of one obstacle
            #     (ox1, oy1, None, None),
            #     (ox2, oy2, None, None),
            #     (ox1, oy2, None, None),
            #     (ox2, oy1, None, None)
            # ],
            # "obstacle-edges": [  # all edges of one obstacle
            #     (ox1, oy1+(oy2-oy1)/2, None, None),
            #     (ox2, oy1+(oy2-oy1)/2, None, None),
            #     (ox1+(ox2-ox1)/2, oy1, None, None),
            #     (ox1+(ox2-ox1)/2, oy2, None, None)
            # ],
        }

        for postype, pos in positions.items():
            print("POSTYPE:", postype)

            plotdir = os.path.join(locs.logs, f"{postype}")
            if not os.path.exists(plotdir):
                os.mkdir(plotdir)

            for i, (x_, y_, xd, yd) in enumerate(pos):

                # leave untouched
                tolerance = .3
                tolerance_ = 1.5

                pdfvars = {}

                if x_ is not None:
                    pdfvars['x_in'] = ContinuousSet(x_ - tolerance_, x_ + tolerance_)

                if y_ is not None:
                    pdfvars['y_in'] = ContinuousSet(y_ - tolerance_, y_ + tolerance_)

                if xd is not None:
                    pdfvars['xdir_in'] = ContinuousSet(xd - tolerance, xd + tolerance)

                if yd is not None:
                    pdfvars['ydir_in'] = ContinuousSet(yd - tolerance, yd + tolerance)

                pdfvars = {}
                # pdfvars['x_in'] = ContinuousSet(99.9, 100)
                # pdfvars['y_in'] = ContinuousSet(0, 100)

                # generate tree conditioned on given position and/or direction
                cond = j.conditional_jpt(
                    evidence=j.bind(
                        {k: v for k, v in pdfvars.items() if k in j.varnames},
                        allow_singular_values=False
                    ),
                    fail_on_unsatisfiability=False
                )

                post = j.posterior(evidence=j.bind(
                        {k: v for k, v in pdfvars.items() if k in j.varnames},
                        allow_singular_values=False
                    ),
                    fail_on_unsatisfiability=False)

                print(len(j.allnodes), len(cond.allnodes))

                # data generation
                x = np.linspace(*limx, 50)
                y = np.linspace(*limy, 50)

                X, Y = np.meshgrid(x, y)
                Z = np.array(
                    [
                        cond.pdf(
                            cond.bind(
                                {
                                    'x_out': x,
                                    'y_out': y
                                }
                            )
                        ) for x, y, in zip(X.ravel(), Y.ravel())
                    ]
                ).reshape(X.shape)
                lbl = np.full(Z.shape, '<br>'.join([f'{vname}: {val}' for vname, val in pdfvars.items()]))

                data = pd.DataFrame(
                    data=[[x, y, Z, lbl]],
                    columns=['x', 'y', 'z', 'lbl']
                )

                prefix = f'POS({x_:{"+.1f" if x_ is not None else ""}},{y_:{"+.1f" if y_ is not None else ""}})_DIR({xd:{"+.1f" if xd is not None else ""}},{yd:{"+.1f" if yd is not None else ""}})'

                # plot ground truth
                plot_data_subset(
                    df,
                    xvar='x_out',
                    yvar='y_out',
                    constraints=pdfvars,
                    limx=limx,
                    limy=limy,
                    save=os.path.join(plotdir, f"{prefix}-gt.svg"),
                    show=False
                )

                # plot JPT Heatmap
                plot_heatmap(
                    xvar='x',
                    yvar='y',
                    data=data,
                    title=None,  # f'pdf({",".join([f"{vname}: {val}" for vname, val in pdfvars.items()])})',
                    limx=limx,
                    limy=limy,
                    show=False,
                    save=os.path.join(plotdir, f"{prefix}-dist-hm.svg"),
                    showbuttons=False
                )

                # plot JPT 3D-Surface
                plot_heatmap(
                    xvar='x',
                    yvar='y',
                    data=data,
                    title=f'pdf({",".join([f"{vname}: {val}" for vname, val in pdfvars.items()])})',
                    limx=limx,
                    limy=limy,
                    show=False,
                    save=os.path.join(plotdir, f"{prefix}-dist-surface.html"),
                    fun="surface"
                )

    def test_reproduce_data_turn(self) -> None:
        # load data and JPT that has been learnt from this data
        j = ThesisPlotsTests.models['000-robotaction_turn.tree']
        df = pd.read_parquet(os.path.join(ThesisPlotsTests.recent, 'data', f'000-robotaction_turn.parquet'))

        # set settings
        limx = (-1, 1)
        limy = (-1, 1)

        # constraints/query values
        dirs = {
            "no-dir": [
                (None, None, None),
                (0, None, None),
                (None, 0, None)
            ],
            "dir": [  # all directions
                (-1, None, None),
                (-1, 0, None),
                (1, None, None),
                (1, 0, None),
                (-.5, None, None),
                (-.5, .5, None),
                (-.5, -.5, None),
                (.5, None, None),
                (.5, .5, None),
                (.5, -.5, None),
                (0, 1, None),
                (0, -1, None),
                (None, 1, None),
                (None, .5, None),
                (None, -1, None),
                (None, -.5, None),
            ],
            "angle": [
                (-1, None, None),

            ]
        }

        for dirtype, pos in dirs.items():

            plotdir = os.path.join(locs.logs, f"{dirtype}")
            if not os.path.exists(plotdir):
                os.mkdir(plotdir)

            for i, (xd, yd, angle) in enumerate(pos):

                # leave untouched
                tolerance = .3
                tolerance_ = 3

                pdfvars = {}

                if xd is not None:
                    pdfvars['xdir_in'] = ContinuousSet(xd - tolerance, xd + tolerance)

                if yd is not None:
                    pdfvars['ydir_in'] = ContinuousSet(yd - tolerance, yd + tolerance)

                if angle is not None:
                    pdfvars['angle'] = ContinuousSet(angle - tolerance_, angle + tolerance_)

                print("PDFVARS:", pdfvars)

                # generate tree conditioned on given position and/or direction
                cond = j.conditional_jpt(
                    evidence=j.bind(
                        {k: v for k, v in pdfvars.items() if k in j.varnames},
                        allow_singular_values=False
                    ),
                    fail_on_unsatisfiability=False
                )

                # data generation
                x = np.linspace(*limx, 50)
                y = np.linspace(*limy, 50)

                X, Y = np.meshgrid(x, y)
                Z = np.array(
                    [
                        cond.pdf(
                            cond.bind(
                                {
                                    'xdir_out': x,
                                    'ydir_out': y
                                }
                            )
                        ) for x, y, in zip(X.ravel(), Y.ravel())
                    ]
                ).reshape(X.shape)
                lbl = np.full(Z.shape, '<br>'.join([f'{vname}: {val}' for vname, val in pdfvars.items()]))

                data = pd.DataFrame(
                    data=[[x, y, Z, lbl]],
                    columns=['x', 'y', 'z', 'lbl']
                )

                prefix = f'DIR({xd:{"+.1f" if xd is not None else ""}},{yd:{"+.1f" if yd is not None else ""}})_{angle:{"+.1f" if angle is not None else ""}}°'

                # plot JPT Heatmap
                plot_heatmap(
                    xvar='x',
                    yvar='y',
                    data=data,
                    title=f'pdf({",".join([f"{vname}: {val}" for vname, val in pdfvars.items()])})',
                    limx=limx,
                    limy=limy,
                    show=False,
                    save=os.path.join(plotdir, f"{prefix}-dist-hm.svg")
                )

                # plot JPT 3D-Surface
                plot_heatmap(
                    xvar='x',
                    yvar='y',
                    data=data,
                    title=f'pdf({",".join([f"{vname}: {val}" for vname, val in pdfvars.items()])})',
                    limx=limx,
                    limy=limy,
                    show=False,
                    save=os.path.join(plotdir, f"{prefix}-dist-surface.html"),
                    fun="surface"
                )

                # plot ground truth
                plot_data_subset(
                    df,
                    xvar='xdir_out',
                    yvar='ydir_out',
                    constraints=pdfvars,
                    limx=limx,
                    limy=limy,
                    save=os.path.join(plotdir, f"{prefix}-ground-truth.svg"),
                    show=False
                )

    def test_astar_cram_path(self) -> None:
        initx, inity, initdirx, initdiry = [20, 70, -1, 0]
        shift = False
        tolerance = .01

        dx = Gaussian(initx, tolerance).sample(500)
        distx = Numeric()
        distx.fit(dx.reshape(-1, 1), col=0)

        dy = Gaussian(inity, tolerance).sample(500)
        disty = Numeric()
        disty.fit(dy.reshape(-1, 1), col=0)

        ddx = Gaussian(initdirx, tolerance).sample(500)
        distdx = Numeric()
        distdx.fit(ddx.reshape(-1, 1), col=0)

        ddy = Gaussian(initdiry, tolerance).sample(500)
        distdy = Numeric()
        distdy.fit(ddy.reshape(-1, 1), col=0)

        initstate = State_()
        initstate.update(
            {
                'x_in': distx,
                'y_in': disty,
                'xdir_in': distdx,
                'ydir_in': distdy
            }
        )

        cmds = [
            # {'tree': '000-robotaction_turn.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-15, -12)}},
            {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-robotaction_turn.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-15, -12)}},
            {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-robotaction_turn.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -9)}},
            {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-robotaction_turn.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(15, 17)}},
            {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-robotaction_turn.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-12, -10)}},
            {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-robotaction_turn.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-5, -3)}},
            {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-robotaction_turn.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(3, 5)}},
            {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-robotaction_turn.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(15, 16)}},  # STOP
            # {'tree': '000-robotaction_turn.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            # {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-robotaction_turn.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-14, -10)}},
            # {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-robotaction_turn.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-20, -18)}},
            # {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-robotaction_turn.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            # {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-robotaction_turn.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-14, -10)}},
            # {'tree': '000-robotaction_turn.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-20, -10)}},
            # {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-robotaction_turn.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            # {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-robotaction_turn.tree', 'params': {'action': 'turn', 'angle': ContinuousSet(-10, -8)}},
            # {'tree': '000-robotaction_turn.tree', 'params': {'action': 'turn', 'angle':  ContinuousSet(-8, -3)}},
            # {'tree': '000-robotaction_move.tree', 'params': {'action': 'move'}}
        ]

        # VARIANT II: each leaf of the conditional tree represents one possible action
        s = initstate
        p = [[s, {}]]
        for i, cmd in enumerate(cmds):
            print(f'Step {i} of {len(cmds)}: {cmd["params"]["action"]}({cmd["params"].get("angle", "")})')
            t = ThesisPlotsTests.models[cmd['tree']]

            # generate evidence by using intervals from the 5th percentile to the 95th percentile for each distribution
            evidence = {
                var: ContinuousSet(s[var].ppf(.05), s[var].ppf(.95)) for var in s.keys() if var != 'collided'
            }

            if cmd["params"] is not None:
                evidence.update(cmd["params"])

            # candidate is the conditional tree
            # t_ = self.generate_steps(evidence, t)
            best = t.posterior(
                variables=t.targets,
                evidence=t.bind({k: v for k, v in evidence.items() if k in t.varnames},
                    allow_singular_values=False
                ),
                fail_on_unsatisfiability=False
            )

            if best is None:
                print('skipping command', cmd, 'unsatisfiable!')
                continue

            # create successor state
            s_ = State_()
            s_.update({k: v for k, v in s.items()})
            s_.tree = cmd['tree']
            s_.leaf = None

            # update belief state of potential predecessor
            for vn, d in best.items():
                outvar = vn.name
                invar = vn.name.replace('_out', '_in')
                if outvar != invar and invar in s_:
                    # if the _in variable is already contained in the state, update it by adding the delta
                    # from the leaf distribution
                    if shift:
                        s_[invar] = Numeric().set(
                            QuantileDistribution.from_cdf(s_[invar].cdf.xshift(-best[outvar].expectation())))
                    else:
                        if len(s_[invar].cdf.functions) > 20:
                            s_[invar] = s_[invar].approximate(n_segments=20)
                            # s_[invar] = s_[invar].approximate_fast(eps=.01)
                        if len(best[outvar].cdf.functions) > 20:
                            best[outvar] = best[outvar].approximate(n_segments=20)
                            # best[outvar] = best[outvar].approximate_fast(eps=.01)
                        s_[invar] = s_[invar] + best[outvar]
                else:
                    s_[invar] = d

                if not shift:
                    if hasattr(s_[invar], 'approximate'):
                        s_[invar] = s_[invar].approximate(n_segments=20)
                        # s_[invar] = s_[invar].approximate_fast(eps=.01)

            p.append([s_, cmd['params']])
            s = State_()
            s.update({k: v for k, v in s_.items()})

        # plot annotated rectangles representing the obstacles and world boundaries
        obstacles = [
            ((0, 0, 100, 100), "kitchen_boundaries"),
            ((15, 10, 25, 20), "chair1"),
            ((35, 10, 45, 20), "chair2"),
            ((10, 30, 50, 50), "kitchen_island"),
            ((80, 30, 100, 70), "stove"),
            ((10, 80, 50, 100), "kitchen_unit"),
            ((60, 80, 80, 100), "fridge"),
        ]

        # plot path as scatter points with direction arrows in kitchen world
        fig = plot_path(
            'x_in',
            'y_in',
            p,
            save=os.path.join(locs.logs, f'crampath.svg'),
            obstacles=obstacles,
            show=False
        )

        fig.write_html(
            os.path.join(locs.logs, f'crampath.html'),
            config=defaultconfig,
            include_plotlyjs="cdn"
        )

        fig.show(config=defaultconfig)

        # plot animation of heatmap representing position distribution update
        plot_pos(
            path=p,
            save=os.path.join(locs.logs, f'crampath-animation.html'),
            show=True,
            limx=(0, 100),
            limy=(0, 100)
        )

        # plot animation of collision bar chart representing change of collision status
        # frames = [s['collided'].plot(view=False).data for (s, _) in p if 'collided' in s]
        # plotly_animation(
        #     data=frames,
        #     save=os.path.join(locs.logs, f'collision.html'),
        #     show=True
        # )

        plot_dir(
            path=p,
            save=os.path.join(locs.logs, f'dirxy.html'),
            show=True,
            limx=(0, 100),
            limy=(0, 100)
        )

        # plot_xyvars(
        #     xvar='x_in',
        #     yvar='y_in',
        #     path=p,
        #     title=f'Position(x,y)',
        #     limx=[-75, -25],
        #     limy=[40, 75],
        #     limz=[0, 0.05],
        #     save=f'test_astar_cram_path_posxy',
        #     show=False
        # )

    def test_move_till_collision(self) -> None:
        print("loading example", ThesisPlotsTests.recent_move)

        initx, inity, initdirx, initdiry = [10, 70, -1, 0]
        tolerance = .01

        dx = Gaussian(initx, tolerance).sample(500)
        distx = Numeric()
        distx.fit(dx.reshape(-1, 1), col=0)

        dy = Gaussian(inity, tolerance).sample(500)
        disty = Numeric()
        disty.fit(dy.reshape(-1, 1), col=0)

        ddx = Gaussian(initdirx, tolerance).sample(500)
        distdx = Numeric()
        distdx.fit(ddx.reshape(-1, 1), col=0)

        ddy = Gaussian(initdiry, tolerance).sample(500)
        distdy = Numeric()
        distdy.fit(ddy.reshape(-1, 1), col=0)

        initstate = State_()
        initstate.update(
            {
                'x_in': distx,
                'y_in': disty,
                'xdir_in': distdx,
                'ydir_in': distdy
            }
        )

        # VARIANT II: each leaf of the conditional tree represents one possible action
        s = initstate
        p = [[s, {}]]
        t = ThesisPlotsTests.models['000-robotaction_move.tree']
        for i, step in enumerate(range(10)):
            print(f'Step {i}: move()')

            # generate evidence by using intervals from the 5th percentile to the 95th percentile for each distribution
            evidence = {
                var: ContinuousSet(s[var].ppf(.05), s[var].ppf(.95)) for var in s.keys() if isinstance(s[var], Numeric)
            }

            # candidate is the conditional tree
            best = t.posterior(
                variables=t.targets,
                evidence=t.bind({k: v for k, v in evidence.items() if k in t.varnames},
                    allow_singular_values=False
                ),
                fail_on_unsatisfiability=False
            )

            if best is None:
                print('skipping at step', step, 'unsatisfiable!')
                continue

            # create successor state
            s_ = State_()
            s_.update({k: v for k, v in s.items()})
            s_.tree = '000-robotaction_move.tree'
            s_.leaf = None

            # update belief state of potential predecessor
            print("Updating new state...")
            for vn, d in best.items():
                outvar = vn.name
                invar = vn.name.replace('_out', '_in')
                if outvar != invar and invar in s_:
                    # if the _in variable is already contained in the state, update it by adding the delta
                    # from the leaf distribution
                    if len(s_[invar].cdf.functions) > 20:
                        s_[invar] = s_[invar].approximate(n_segments=20)
                    if len(best[outvar].cdf.functions) > 20:
                        best[outvar] = best[outvar].approximate(n_segments=20)

                    s_[invar] = s_[invar] + best[outvar]
                else:
                    s_[invar] = d

                if hasattr(s_[invar], 'approximate'):
                    s_[invar] = s_[invar].approximate(n_segments=20)

            p.append([s_, {'action': 'move'}])
            s = State_()
            s.update({k: v for k, v in s_.items()})

        # plot annotated rectangles representing the obstacles and world boundaries
        obstacles = [
            ((0, 0, 100, 100), "kitchen_boundaries"),
            ((15, 10, 25, 20), "chair1"),
            ((35, 10, 45, 20), "chair2"),
            ((10, 30, 50, 50), "kitchen_island"),
            ((80, 30, 100, 70), "stove"),
            ((10, 80, 50, 100), "kitchen_unit"),
            ((60, 80, 80, 100), "fridge"),
        ]

        fig = plot_path(
            'x_in',
            'y_in',
            p,
            save=os.path.join(locs.logs, f'crampath-collision.svg'),
            obstacles=obstacles,
            show=False
        )

        fig.write_html(
            os.path.join(locs.logs, f'crampath-collision.html'),
            config=defaultconfig,
            include_plotlyjs="cdn"
        )

        fig.show(config=defaultconfig)

        # print heatmap representing position distribution update
        plot_pos(
            path=p,
            save=os.path.join(locs.logs, f'crampath-collision-animation.html'),
            show=True,
            limx=(0, 100),
            limy=(0, 100)
        )

        # plot animation of collision bar chart representing change of collision status
        frames = [s['collided'].plot(view=False).data for (s, _) in p if 'collided' in s]
        plotly_animation(
            data=frames,
            save=os.path.join(locs.logs, f'collision.html'),
            show=True
        )

    def test_plot_kaleido_error(self) -> None:
        # small values around 0 (i.e. values smaller
        # than abs(num.e-306)) will break plotting of
        # distributions with plotly/kaleido
        X_ = [0.e+000, - 5.e-324,  0.e+000,  0.e+000]
        Y = [1., 0., 1., 1.]

        mainfig = go.Figure()

        # plot dashed CDF
        mainfig.add_trace(
            go.Scatter(
                x=X_,
                y=Y,
                mode='lines',
                name='Piecewise linear CDF from bounds',
                line=dict(
                    color=f'rgba(15,21,110,1.0)',
                    width=4,
                    dash='dash'
                )
            )
        )

        mainfig.update_layout(
            xaxis=dict(
                title='x',
                side='bottom'
            ),
            yaxis=dict(
                title='%'
            ),
            title=f'Distribution'
        )

        mainfig.write_image(
            os.path.join(locs.logs, 'testimg.png'),
            scale=1
        )

    def test_data_point_update_plot(self) -> None:
        def turn(x, y, deg):
            deg = np.radians(-deg)
            return x * math.cos(deg) - y * math.sin(deg), x * math.sin(deg) + y * math.cos(deg)
        
        d = [
            (
                0,
                0,
                *dir,
                f"Step {s}",
                "",
                1
            ) for s, dir in enumerate([
                turn(*(0,1), 10*i) for i in range(36)
            ])
        ]
        
        data = pd.DataFrame(
            data=d,
            columns=['x', 'y', 'dx', 'dy', 'step', 'lbl', 'size']
        )

        fig = go.Figure()
        fig_ = plot_scatter_quiver(
            'x',
            'x',
            data,
            title=None,
            save=os.path.join(locs.logs, 'star.svg'),
        )

        fig.add_traces(fig_.data)
        fig.layout = fig_.layout
        fig.update_layout(
            height=1000,
            width=1000,
            title=None,
            xaxis=dict(
                title=xvar,
                side='top',
                range=[*limx]
            ),
            yaxis=dict(
                title=yvar,
                range=[*limy]
            )
        )

        fig.show(config=defaultconfig)

    def tearDown(self) -> None:
        # draw path steps into grid (use action symbols)
        print()

    @classmethod
    def tearDownClass(cls) -> None:
        print()
