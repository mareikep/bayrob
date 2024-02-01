import math
import os
import traceback
import unittest

import unittest
from functools import reduce
from pathlib import Path

import dnutils
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from pandas import DataFrame

from calo.core.astar_jpt import State
from calo.utils import locs
from calo.utils.plotlib import plot_heatmap, plot_data_subset, plot_tree_dist, plot_pos, plot_path, defaultconfig, \
    plotly_animation, plot_scatter_quiver, plot_dir, filter_dataframe, plot_multiple_dists, fig_to_file, plotly_sq, \
    plot_tree_leaves
from calo.utils.utils import recent_example, fmt
from jpt import SymbolicType, NumericVariable, JPT
from jpt.base.intervals import ContinuousSet, RealSet

from jpt.base.functions import PiecewiseFunction
from jpt.distributions import Gaussian, Numeric
from jpt.distributions.quantile.quantiles import QuantileDistribution


logger = dnutils.getlogger("thesis_tests", level=dnutils.DEBUG)


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
       cls.recent_move = recent_example(os.path.join(locs.examples, 'move'))
       cls.recent_movemini = recent_example(os.path.join(locs.examples, 'move_exp'))
       cls.recent_turn = recent_example(os.path.join(locs.examples, 'turn'))
       cls.recent_perception = recent_example(os.path.join(locs.examples, 'perception'))
       cls.recent_pr2 = recent_example(os.path.join(locs.examples, 'pr2'))
       print(f"loading examples from:")
       print(cls.recent_move)
       print(cls.recent_turn)
       print(cls.recent_perception)
       print(cls.recent_pr2)

       cls.models = dict(
           [
               (
                   treefile.name,
                   JPT.load(str(treefile))
               )
               for p in [cls.recent_move, cls.recent_turn, cls.recent_perception, cls.recent_pr2]
               for treefile in Path(p).rglob('*.tree')
           ]
       )

       cls.obstacle_chair1 = [15, 10, 25, 20]  # chair1"
       cls.obstacle_chair2 = [35, 10, 45, 20]  # "chair2"
       cls.obstacle_kitchen_island = [10, 30, 50, 50]  # "kitchen_island"
       cls.obstacle_stove = [80, 30, 100, 70]  # "stove"
       cls.obstacle_kitchen_unit = [10, 80, 50, 100]  # "kitchen_unit"
       cls.obstacle_fridge = [60, 80, 80, 100]  # "fridge"

       cls.obstacles = [
           ((0, 0, 100, 100), "kitchen_boundaries"),
           (cls.obstacle_chair1, "chair1"),
           (cls.obstacle_chair2, "chair2"),
           (cls.obstacle_kitchen_island, "kitchen_island"),
           (cls.obstacle_stove, "stove"),
           (cls.obstacle_kitchen_unit, "kitchen_unit"),
           (cls.obstacle_fridge, "fridge"),
       ]

    def test_plot_init_dist(self) -> None:
        # plot initial position (in) distribution of move tree
        t = self.models['000-move.tree']
        plot_tree_dist(
            tree=t,
            qvarx=t.varnames['x_in'],
            qvary=t.varnames['y_in'],
            title='Initial distribution P(x,y)',
            limx=(-100, 100),
            limy=(-100, 100),
            save=os.path.join(os.path.join(locs.logs, 'test_plot_init_dist.html')),
            show=True
        )

    def test_plot_dist_similarity_discrete(self) -> None:
        # plot for explaining similarity of discrete dists
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
            os.path.join(locs.logs, f'test_plot_dist_similarity_discrete.svg'),
            scale=1
        )

        mainfig.write_html(
            os.path.join(locs.logs, f'test_plot_dist_similarity_discrete.html'),
            config=self.defaultconfig,
            include_plotlyjs="cdn"
        )

        mainfig.show(
            config=self.defaultconfig
        )

        # Assert

    def test_plot_dist_add_continuous(self) -> None:
        # plot for explaining addition of continuous dist
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

        fig_to_file(mainfig, os.path.join(locs.logs, f'test_plot_dist_add_continuous.html'), ftypes=['.svg', '.html'])

        mainfig.show(
            config=defaultconfig('test_plot_dist_similarity_continuous')
        )

    def test_plot_dist_similarity_continuous(self) -> None:
        # plot for explaining similarity of continuous dists
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
                name=f'$X_1 \sim D_1; D2 = \cal{{N}}({mu1},{v1})$',
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

        fig_to_file(mainfig, os.path.join(locs.logs, f'test_plot_dist_similarity_continuous.html'), ftypes=['.svg', '.html'])

    def test_plot_dist_similarity_continuous_3(self) -> None:
        # plot for explaining similarity of continuous dists
        # Arrange

        limx = (-5, 15)
        mu1, mu2, mu3 = [-2, 3, 12]
        v1, v2, v3 = [.4, .4, .4]
        gauss1 = Gaussian(mu1, v1)
        gauss2 = Gaussian(mu2, v2)
        gauss3 = Gaussian(mu3, v3)

        x = np.linspace(*limx, 300)
        pdfg1 = gauss1.pdf(x)
        pdfg2 = gauss2.pdf(x)
        pdfg3 = gauss3.pdf(x)

        # Act
        mainfig = go.Figure()

        mainfig.add_trace(
            go.Scatter(
                x=x,
                y=pdfg1,
                mode='lines',
                name=f'$X_1 \sim d_1; d_1 = \cal{{N}}({mu1},{v1})$',
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
                name=f'$X_2 \sim d_2; d_2 = \cal{{N}}({mu2},{v2})$',
                line=dict(
                    color='rgba(134, 129, 177,1)',
                    width=4,
                )
            )
        )

        mainfig.add_trace(
            go.Scatter(
                x=x,
                y=pdfg3,
                mode='lines',
                name=f'$X_d \sim d_3; d_3 = \cal{{N}}({mu3},{v3})$',
                line=dict(
                    color='rgba(138, 203, 183,1)',
                    width=4,
                    # dash="dot"
                )
            )
        )

        mainfig.update_layout(
            xaxis=dict(
                title='$x$',
                range=limx
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
            ),
            height=1000,
            width=1000,
        )

        fig_to_file(mainfig, os.path.join(locs.logs, f'test_plot_dist_similarity_continuous_3.html'), ftypes=['.svg', '.html'])

    def test_plot_dist_similarity_continuous_wasserstein(self) -> None:
        # plot for explaining similarity of continuous dists
        # Arrange

        limx = (-5, 15)
        mu1, mu2, mu3 = [-2, 3, 12]
        v1, v2, v3 = [.4, .4, .4]
        gauss1 = Gaussian(mu1, v1)
        gauss2 = Gaussian(mu2, v2)
        gauss3 = Gaussian(mu3, v3)

        x = np.linspace(*limx, 300)

        # generate dists to
        d1 = gauss1.sample(500)
        dist1 = Numeric()
        dist1.fit(d1.reshape(-1, 1), col=0)
        figd1 = dist1.plot(view=False, color='rgba(0,104,180,1)', xlabel='x', title=f'$X_1 \sim D_2; D2 = \cal{{N}}({mu1},{v1})$')

        d2 = gauss2.sample(500)
        dist2 = Numeric()
        dist2.fit(d2.reshape(-1, 1), col=0)
        figd2 = dist2.plot(view=False, color='rgba(134, 129, 177,1)', xlabel='x', title=f'$X_2 \sim D_2; D2 = \cal{{N}}({mu2},{v2})$')

        d3 = gauss3.sample(500)
        dist3 = Numeric()
        dist3.fit(d3.reshape(-1, 1), col=0)
        figd3 = dist3.plot(view=False, color='rgba(138, 203, 183,1)', xlabel='x', title=f'$X_3 \sim D_1; D1 = \cal{{N}}({mu3},{v3})$')

        diff1 = PiecewiseFunction.abs(dist1.cdf - dist2.cdf)
        dist4 = Numeric().set(params=QuantileDistribution.from_cdf(diff1))
        figd4 = dist4.plot(view=False, color='#383838', xlabel='x', title=f'$d_3 | d_1 - d_2 |$')
        figd4.data[0].update(dict(
            fill='tozeroy',
            fillcolor="#d3d3d3",
            fillpattern=dict(shape='/'),
            line=dict(color="#383838"),
            name="diff1",
            opacity=0.5
        ))

        diff2 = PiecewiseFunction.abs(dist2.cdf - dist3.cdf)
        dist5 = Numeric().set(params=QuantileDistribution.from_cdf(diff2))
        figd5 = dist5.plot(view=False, color='#383838', xlabel='x', title=f'$d_3 | d_1 - d_2 |$')
        figd5.data[0].update(dict(
            fill='tozeroy',
            fillcolor="#d3d3d3",
            fillpattern=dict(shape='x'),
            line=dict(color="#383838"),
            name="diff2",
            opacity=0.5
        ))

        d12 = dist1.distance(dist2)
        d13 = dist1.distance(dist3)
        d23 = dist2.distance(dist3)

        s12 = dist1.similarity(dist2)
        s13 = dist1.similarity(dist3)
        s23 = dist2.similarity(dist3)

        print('Wasserstein', d12, d13, d23)
        print('Jaccard', s12, s13, s23)

        # Act
        mainfig = go.Figure()

        mainfig.add_traces(figd4.data)
        mainfig.add_traces(figd5.data)
        mainfig.add_traces(figd1.data)
        mainfig.add_traces(figd2.data)
        mainfig.add_traces(figd3.data)

        # mainfig.add_trace(
        #     go.Scatter(
        #         x=x,
        #         y=pdfg1,
        #         mode='lines',
        #         name=f'$X_1 \sim \cal{{N}}({mu1},{v1})$',
        #         line=dict(
        #             color='rgba(0,104,180,1)',
        #             width=4,
        #         )
        #     )
        # )
        #
        # mainfig.add_trace(
        #     go.Scatter(
        #         x=x,
        #         y=pdfg2,
        #         mode='lines',
        #         name=f'$X_2 \sim \cal{{N}}({mu2},{v2})$',
        #         line=dict(
        #             color='rgba(134, 129, 177,1)',
        #             width=4,
        #         )
        #     )
        # )
        #
        # mainfig.add_trace(
        #     go.Scatter(
        #         x=x,
        #         y=pdfg3,
        #         mode='lines',
        #         name=f'$X_3 \sim \cal{{N}}({mu3},{v3})$',
        #         line=dict(
        #             color='rgba(138, 203, 183,1)',
        #             width=4,
        #             # dash="dot"
        #         )
        #     )
        # )

        mainfig.update_layout(
            xaxis=dict(
                title='$x$',
                range=limx
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
            ),
            height=1000,
            width=1000,
        )
        fig_to_file(mainfig, os.path.join(locs.logs, f'test_plot_dist_similarity_continuous_wasserstein.html'), ftypes=['.svg', '.html'])

    def test_reproduce_data_find_limits(self) -> None:
        # for constrained MOVE TARGET variables, plot heatmap, 3D and ground data of position (IN) distribution

        # load data and JPT that has been learnt from this data
        j = self.models['000-move.tree']
        print(f"Loading tree from {self.recent_move}")

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
            # 'x_out': ContinuousSet(xoutmin, xoutmax),
            # 'y_out': ContinuousSet(youtmin, youtmax),
            'x_out': ContinuousSet(np.NINF, np.PINF),
            'y_out': ContinuousSet(np.NINF, np.PINF),
            # 'collided': True
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
        dist = plot_heatmap(
            xvar='x',
            yvar='y',
            data=data,
            limx=limx,
            limy=limy,
            limz=(0, 0.0002),
            save=None,
            show=True,
        )
        fig_to_file(dist, os.path.join(locs.logs, f"test_reproduce_data_find_limits-{prefix}-dist-hm.html"), ftypes=['.svg', '.html'])

        # plot ground truth
        df = pd.read_parquet(os.path.join(self.recent_move, 'data', f'000-move.parquet'))
        gt = plot_data_subset(
            df,
            xvar="x_in",
            yvar="y_in",
            constraints=pdfvars,
            limx=limx,
            limy=limy,
            save=None,
            show=True,
            color='rgb(0,104,180)'
        )
        fig_to_file(gt, os.path.join(locs.logs, f"test_reproduce_data_find_limits-{prefix}-gt.html"), ftypes=['.svg', '.html'])

    def test_reproduce_data_single_jpt(self) -> None:
        # SINGLE set of constraints:
        # for any constrained MOVE variables, plot heatmap, 3D and ground data of position (OUT) distribution

        # load data and JPT that has been learnt from this data
        j = self.models['000-move.tree']
        print(f"Loading tree from {self.recent_move}")

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
        df = pd.read_parquet(os.path.join(self.recent_move, 'data', f'000-move.csv'))
        plot_data_subset(
            df,
            "x_out",
            "y_out",
            pdfvars,
            limx=limx,
            limy=limy,
            show=True,
            color='rgb(0,104,180)'
        )

    def test_face(self) -> None:
        # plot ground truth and distribution smiley face
        j = self.models['000-move.tree']
        df = pd.read_parquet(os.path.join(self.recent_move, 'data', f'000-move.parquet'))
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
        # plot_data_subset(
        #     df,
        #     xvar='x_out',
        #     yvar='y_out',
        #     constraints={},
        #     limx=limx,
        #     limy=limy,
        #     show=True,
        #     color = 'rgb(0,104,180)'
        # )

        from jpt import infer_from_dataframe
        variables = infer_from_dataframe(
            df,
            scale_numeric_types=False,
            # precision=.5
        )

        jpt_ = JPT(
            variables=variables,
            targets=variables[4:],
            min_impurity_improvement=None,
            min_samples_leaf=400
        )

        jpt_.learn(df, close_convex_gaps=False)

        # jpt_.save(os.path.join(self.recent_move, f'000-funnyface.tree'))
        #
        # jpt_.plot(
        #     title='funnyface',
        #     plotvars=list(jpt_.variables),
        #     filename=f'000-funnyface',
        #     directory=os.path.join(self.recent_move, 'plots'),
        #     leaffill='#CCDAFF',
        #     nodefill='#768ABE',
        #     alphabet=True,
        #     view=False
        # )

        # generate tree conditioned on given position and/or direction
        # jpt_ = j.conditional_jpt(
        #     evidence=j.bind(
        #         {
        #             'xdir_in': rx,
        #             'ydir_in': ry
        #         },
        #         allow_singular_values=False
        #     ),
        #     fail_on_unsatisfiability=False
        # )

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
            save=os.path.join(locs.logs, f"test_reproduce_data_single_jpt.html"),
            showbuttons=True
        )

    def test_reproduce_data_move(self) -> None:
        # MULTIPLE sets of constraints:
        # for constrained MOVE FEATURE variables, plot heatmap, 3D and ground data of position (OUT) distribution
        mini = True
        addobstacles = True
        stepsize = 1
        
        if mini:
            # load data and JPT that has been learnt from this data
            j = JPT.load(os.path.join(self.recent_movemini, '000-move_exp.tree'))
            j.postprocess_leaves()
            j = j.prune(.77)
            df = pd.read_parquet(os.path.join(self.recent_movemini, 'data', f'000-move_exp.parquet'))
            modelpath = self.recent_movemini
            xl, yl, xu, yu = (0, 0, 10, 10)
            oxl, oyl, oxu, oyu = (5, 5, 7, 7)
            freepos = 8
        else:
            j = self.models['000-move.tree']
            df = pd.read_parquet(os.path.join(self.recent_move, 'data', f'000-move.parquet'))
            modelpath = self.recent_move
            xl, yl, xu, yu = (0, 0, 100, 100)
            oxl, oyl, oxu, oyu = self.obstacle_kitchen_island
            freepos = 60

        # constraints/query values (x_in, y_in, xdir_in, ydir_in)
        positions = {
            "grid-corners": [  # all corners of gridworld
                (ContinuousSet(xl, xl + stepsize), ContinuousSet(yl, yl + stepsize), None, None, {}),  # lower left
                (ContinuousSet(xl, xl + stepsize), ContinuousSet(yu - stepsize, yu), None, None, {}),  # upper left
                (ContinuousSet(xu - stepsize, xu), ContinuousSet(yl, yl + stepsize), None, None, {}),  # lower right
                (ContinuousSet(xu - stepsize, xu), ContinuousSet(yu - stepsize, yu), None, None, {})  # upper right
            ],
            "grid-edges": [  # all edges of gridworld (center)
                (xl, None, -1, 0, {}),  # left edge
                (xl, None, 1, 0, {}),  # left edge
                (xl, None, None, None, {}),  # left edge
                (xu, None, None, None, {}),  # right edge
                (None, yl, None, None, {}),  # lower edge
                (None, yu, None, None, {})  # upper edge
            ],
            # "apriori": [
            #     (None, None, None, None, {}),
            # ],
            # "in": [
            #     (None, None, None, None, {"collided": True}),
            #     (None, None, None, None, {"x_out": 0, "y_out": 0}),
            #     (None, None, None, None, {}),
            #     (ContinuousSet(xl, xl + stepsize), ContinuousSet(yl, yl + stepsize), None, None, {}),  # lower left
            #     (ContinuousSet(xl, xl + stepsize), ContinuousSet(yu - stepsize, yu), None, None, {}),  # upper left
            #     (ContinuousSet(xu - stepsize, xu), ContinuousSet(yl, yl + stepsize), None, None, {}),  # lower right
            #     (ContinuousSet(xu - stepsize, xu), ContinuousSet(yu - stepsize, yu), None, None, {})  # upper right
            # ],
            # "obstacle-corners": [  # all corners of one obstacle
            #     (oxl, oxl, None, None, {}),  # lower left
            #     (oxl, oyu, None, None, {}),  # upper left
            #     (oxu, oyl, None, None, {}),  # lower right
            #     (oxu, oyu, None, None, {})  # upper right
            # ],
            # "obstacle-edges": [  # all edges of one obstacle
            #     (oxl, ContinuousSet(oyl, oyu), None, None, {}),  # left edge
            #     (oxu, ContinuousSet(oyl, oyu), None, None, {}),  # right edge
            #     (ContinuousSet(oxl, oxu), oyl, None, None, {}),  # lower edge
            #     (ContinuousSet(oxl, oxu), oyu, None, None, {})  # upper edge
            # ],
            # "free-pos": [  # random position in obstacle-free area
            #     (ContinuousSet(freepos - 2, freepos + 2), ContinuousSet(freepos - 2, freepos + 2), -.7, -.7, {}),  # 20, 70 pos (orig)
            #     (ContinuousSet(freepos - 2, freepos + 2), ContinuousSet(freepos - 2, freepos + 2), .7, -.7, {}),
            #     (ContinuousSet(freepos - 2, freepos + 2), ContinuousSet(freepos - 2, freepos + 2), .7, .7, {}),
            #     (ContinuousSet(freepos - 2, freepos + 2), ContinuousSet(freepos - 2, freepos + 2), -.7, .7, {}),
            #     (ContinuousSet(freepos - 2, freepos + 2), ContinuousSet(freepos - 2, freepos + 2), -1, 0, {}),
            #     (ContinuousSet(freepos - 2, freepos + 2), ContinuousSet(freepos - 2, freepos + 2), 1, 0, {}),
            #     (ContinuousSet(freepos - 2, freepos + 2), ContinuousSet(freepos - 2, freepos + 2), 0, 1, {}),
            #     (ContinuousSet(freepos - 2, freepos + 2), ContinuousSet(freepos - 2, freepos + 2), 0, -1, {}),
            #     (ContinuousSet(freepos - 2, freepos + 2), ContinuousSet(freepos - 2, freepos + 2), None, None, {}),  # 50, 50 pos (orig)
            # ],
            # "no-pos": [  # all directions without given pos
            #     (None, None, 0, -1, {}),
            #     (None, None, 0, 1, {}),
            #     (None, None, 1, 0, {}),
            #     (None, None, -1, 0, {}),
            #     (None, None, .5, -.5, {}),
            #     (None, None, .5, .5, {}),
            #     (None, None, -.5, -.5, {}),
            #     (None, None, -.5, .5, {}),
            #     (None, None, -1, None, {}),
            #     (None, None, 1, None, {}),
            #     (None, None, None, -1, {}),
            #     (None, None, None, 1, {}),
            # ]
        }

        for postype, pos in positions.items():
            logger.warning(f"POSTYPE: {postype}")

            plotdir = os.path.join(locs.logs, f"Move-{Path(modelpath).stem}", f"test_reproduce_data_move-{postype}")
            if not os.path.exists(os.path.join(locs.logs, f"Move-{Path(modelpath).stem}")):
                os.mkdir(os.path.join(locs.logs, f"Move-{Path(modelpath).stem}"))
            if not os.path.exists(plotdir):
                os.mkdir(plotdir)

            # set settings
            limx = (xl, xu) if postype == "in" else (-2, 2)
            limy = (xl, xu) if postype == "in" else (-2, 2)

            for i, (x_, y_, xd, yd, more) in enumerate(pos):

                # leave untouched
                tolerance = .3
                tolerance_ = .5

                pdfvars = {}

                if x_ is not None:
                    pdfvars['x_in'] = x_ if isinstance(x_, ContinuousSet) else ContinuousSet(x_ - tolerance_, x_ + tolerance_)

                if y_ is not None:
                    pdfvars['y_in'] = y_ if isinstance(y_, ContinuousSet) else ContinuousSet(y_ - tolerance_, y_ + tolerance_)

                if xd is not None:
                    pdfvars['xdir_in'] = xd if isinstance(xd, ContinuousSet) else ContinuousSet(xd - tolerance, xd + tolerance)

                if yd is not None:
                    pdfvars['ydir_in'] = yd if isinstance(yd, ContinuousSet) else ContinuousSet(yd - tolerance, yd + tolerance)

                if more is not None:
                    pdfvars.update(more)

                logger.info(f"Query: {pdfvars}")
                prefix = f'POS({fmt(x_, prec=1, positive=True)},{fmt(y_, prec=1, positive=True)})_DIR({fmt(xd, prec=1, positive=True)},{fmt(yd, prec=1, positive=True)})[{fmt(more)}]'

                # generate tree conditioned on given position and/or direction
                cond = j.conditional_jpt(
                    evidence=j.bind(
                        {k: v for k, v in pdfvars.items() if k in j.varnames},
                        allow_singular_values=False
                    ),
                    fail_on_unsatisfiability=False
                )

                if cond is None:
                    logger.warning(f"COND IS NONE, skipping {pdfvars}")
                    continue
                logger.info(f"Nodes in original tree: {len(j.allnodes)}; nodes in conditional tree: {len(cond.allnodes)}")

                # plot rectangles representing each leaf participating in this query
                # plot_tree_leaves(
                #     cond,
                #     cond.varnames['x_in'],
                #     cond.varnames['y_in'],
                #     title=prefix,
                #     limx=limx,
                #     limy=limy,
                #     show=True
                # )

                # data generation for distribution plot
                x = np.linspace(*limx, 150)
                y = np.linspace(*limy, 150)

                X, Y = np.meshgrid(x, y)
                from datetime import datetime
                logger.warning(f'Starting to generate data for distribution plot {str(datetime.now())}')
                Z = np.array([cond.pdf(cond.bind({f'x_{"in" if postype == "in" else "out"}': x,f'y_{"in" if postype == "in" else "out"}': y})) for x, y, in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
                logger.warning(f'done generating data for distribution plot {str(datetime.now())}')

                lbl = np.full(Z.shape, '<br>'.join([f'{vname}: {val}' for vname, val in pdfvars.items()]))

                data = pd.DataFrame(
                    data=[[x, y, Z, lbl]],
                    columns=['x', 'y', 'z', 'lbl']
                )

                # plot ground truth
                gt = go.Figure()
                if addobstacles:

                    # plot obstacles in background
                    if self.obstacles is not None and postype == 'in':
                        for (o, on) in self.obstacles:
                            gt.add_trace(
                                plotly_sq(o, lbl=on, color='rgb(59, 41, 106)', legend=False))

                gt_ = plot_data_subset(
                    df,
                    xvar=f'x_{"in" if postype == "in" else "out"}',
                    yvar=f'y_{"in" if postype == "in" else "out"}',
                    constraints=pdfvars,
                    limx=limx,
                    limy=limy,
                    save=None,
                    show=False,
                    color='rgb(0,104,180)'
                )
                if gt_ is not None:
                    gt.layout = gt_.layout
                    gt.add_traces(gt_.data)

                    fig_to_file(gt, os.path.join(plotdir, f"{prefix}-gt.html"), ftypes=['.svg', '.html'])# if postype not in ['in', 'apriori'] else None)
                else:
                    logger.warning("SKIPPING", os.path.join(plotdir, f"{prefix}-gt.html"), "as figure is None")

                dist = go.Figure()
                if addobstacles:

                    # plot obstacles in background
                    if self.obstacles is not None and postype == 'in':
                        for (o, on) in self.obstacles:
                            dist.add_trace(plotly_sq(o, lbl=on, color='rgb(59, 41, 106)', legend=False))

                # plot heatmap
                dist_ = plot_heatmap(
                    xvar='x',
                    yvar='y',
                    data=data,
                    title=f'pdf({",".join([f"{vname}: {val}" for vname, val in pdfvars.items()])})',
                    limx=limx,
                    limy=limy,
                    show=False,
                    save=None,
                    fun="heatmap"
                )

                if dist_ is not None:
                    dist.layout = dist_.layout
                    dist.add_traces(dist_.data)

                fig_to_file(dist, os.path.join(plotdir, f"{prefix}-dist-surface.html"), ftypes=['.svg', '.html'])# if postype not in ['in', 'apriori'] else None)

    def test_reproduce_data_turn(self) -> None:
        # MULTIPLE sets of constraints:
        # for any constrained TURN variables, plot heatmap, 3D and ground data of direction (OUT) distribution

        # load data and JPT that has been learnt from this data
        j = self.models['000-turn.tree']
        df = pd.read_parquet(os.path.join(self.recent_turn, 'data', f'000-turn.parquet'))

        # set settings
        limx = (-1.5, 1.5)
        limy = (-1.5, 1.5)

        # plot init distributions
        initdist = plot_tree_dist(
            tree=j,
            qvarx=j.varnames['xdir_in'],
            qvary=j.varnames['ydir_in'],
            limx=limx,
            limy=limy,
            save=None,
            show=False
        )
        fig_to_file(initdist, os.path.join(locs.logs, f"test_reproduce_data_turn-init_in-dist.html"), ftypes=['.svg', '.html'])
        gt = plot_data_subset(
            df,
            xvar='xdir_in',
            yvar='ydir_in',
            constraints={},
            limx=limx,
            limy=limy,
            save=None,
            show=False,
            color='rgb(0,104,180)'
        )
        fig_to_file(gt, os.path.join(locs.logs, f"test_reproduce_data_turn-init_in-gt.html"), ftypes=['.svg', '.html'])

        initdisto = plot_tree_dist(
            tree=j,
            qvarx=j.varnames['xdir_out'],
            qvary=j.varnames['ydir_out'],
            limx=limx,
            limy=limy,
            save=None,
            show=False
        )
        fig_to_file(initdisto, os.path.join(locs.logs, f"test_reproduce_data_turn-init_out-dist.html"), ftypes=['.svg', '.html'])
        gt = plot_data_subset(
            df,
            xvar='xdir_out',
            yvar='ydir_out',
            constraints={},
            limx=limx,
            limy=limy,
            save=None,
            show=False,
            color='rgb(0,104,180)'
        )
        fig_to_file(gt, os.path.join(locs.logs, f"test_reproduce_data_turn-init_out-gt.html"), ftypes=['.svg', '.html'])

        # constraints/query values (xdir_in, ydir_in, angle)
        dirs = {
            "no-dir": [
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
                (None, None, 45),
                (1, 0, 45),
            ],
            "apriori": [
                (None, None, None),
            ],
            "in": [
                (None, None, None),
            ],
        }

        for dirtype, d in dirs.items():

            plotdir = os.path.join(locs.logs, f"Turn-{Path(self.recent_turn).stem}", f"test_reproduce_data_turn-{dirtype}")
            if not os.path.exists(os.path.join(locs.logs, f"Turn-{Path(self.recent_turn).stem}")):
                os.mkdir(os.path.join(locs.logs, f"Turn-{Path(self.recent_turn).stem}"))
            if not os.path.exists(plotdir):
                os.mkdir(plotdir)

            for i, (xd, yd, angle) in enumerate(d):

                # leave untouched
                tolerance = .3
                tolerance_ = 3

                pdfvars = {}

                if xd is not None:
                    pdfvars['xdir_in'] = xd if isinstance(xd, ContinuousSet) else ContinuousSet(xd - tolerance, xd + tolerance)

                if yd is not None:
                    pdfvars['ydir_in'] = yd if isinstance(yd, ContinuousSet) else ContinuousSet(yd - tolerance, yd + tolerance)

                if angle is not None:
                    pdfvars['angle'] = angle if isinstance(angle, ContinuousSet) else ContinuousSet(angle - tolerance_, angle + tolerance_)

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
                                    f'xdir_{"in" if dirtype == "in" else "out"}': x,
                                    f'ydir_{"in" if dirtype == "in" else "out"}': y
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

                prefix = f'DIR({fmt(xd, prec=1, positive=True)},{fmt(yd, prec=1, positive=True)})_{fmt(angle, prec=1, positive=True)}°'

                # plot ground truth
                gt = plot_data_subset(
                    df,
                    xvar=f'xdir_{"in" if dirtype == "in" else "out"}',
                    yvar=f'ydir_{"in" if dirtype == "in" else "out"}',
                    constraints=pdfvars,
                    limx=limx,
                    limy=limy,
                    save=None,
                    show=False,
                    color='rgb(0,104,180)'
                )
                fig_to_file(gt, os.path.join(plotdir, f"{prefix}-gt.html"), ftypes=['.svg', '.html'])

                # plot distribution
                dist = plot_heatmap(
                    xvar='x',
                    yvar='y',
                    data=data,
                    title=f'pdf({",".join([f"{vname}: {val}" for vname, val in pdfvars.items()])})',
                    limx=limx,
                    limy=limy,
                    show=False,
                    save=None,
                    fun="heatmap"
                )
                fig_to_file(dist, os.path.join(plotdir, f"{prefix}-dist-surface.html"), ftypes=['.svg', '.html'])

    def test_reproduce_data_perception(self) -> None:
        # MULTIPLE sets of constraints:
        # for any constrained PERCEPTION variables, plot all remaining dists

        # load data and JPT that has been learnt from this data
        j = self.models['000-perception.tree']
        addobstacles = True
        df = pd.read_parquet(os.path.join(self.recent_perception, 'data', f'000-perception.parquet'))

        print(f"Loading tree from {self.recent_perception}")
        objects = ['cup', 'cutlery', 'bowl', 'sink', 'milk', 'beer', 'cereal', 'stovetop', 'pot']
        detected_objects = [f'detected({o})' for o in objects]
        containers = ['fridge_door', 'cupboard_door_left', 'cupboard_door_right', 'kitchen_unit_drawer']
        open_containers = [f'open({c})' for c in containers]

        # constraints/query values
        # the postype determines a category, tp
        queries = {
            "milk-detected": [
                ({'detected(milk)': True}, detected_objects + open_containers + ['daytime', 'nearest_furniture', 'positions']),
                ({'detected(milk)': True, 'daytime': ['morning']}, detected_objects + open_containers + ['daytime', 'nearest_furniture', 'positions']),
                ({'detected(milk)': True, 'daytime': ['night']}, detected_objects + open_containers + ['daytime', 'nearest_furniture', 'positions']),
                ({'detected(milk)': True, 'daytime': ['post-breakfast']}, detected_objects + open_containers + ['daytime', 'nearest_furniture', 'positions']),
                ({'detected(milk)': True, 'open(fridge_door)': True, 'daytime': ['night']}, detected_objects + open_containers + ['daytime', 'nearest_furniture', 'positions']),
            ],
            "beer-detected": [
                ({'detected(beer)': True, 'daytime': ['night']}, detected_objects + open_containers + ['daytime', 'nearest_furniture', 'positions']),
            ],
            "bowl-detected": [
                ({'detected(bowl)': True, 'daytime': ['post-breakfast']}, detected_objects + open_containers + ['daytime', 'nearest_furniture', 'positions']),
            ],
            "nearest_furniture": [
                ({'nearest_furniture': 'stove'}, detected_objects + open_containers + ['daytime', 'nearest_furniture', 'positions']),
                ({'nearest_furniture': 'kitchen_unit'}, detected_objects + open_containers + ['daytime', 'nearest_furniture', 'positions']),
                ({'nearest_furniture': 'kitchen_unit', 'open(kitchen_unit_drawer)': True}, detected_objects + open_containers + ['daytime', 'nearest_furniture', 'positions']),
                ({'nearest_furniture': 'kitchen_unit', 'open(cupboard_door_right)': True}, detected_objects + open_containers + ['daytime', 'nearest_furniture', 'positions']),
                ({'nearest_furniture': 'kitchen_unit', 'open(cupboard_door_left)': True}, detected_objects + open_containers + ['daytime', 'nearest_furniture', 'positions']),
                ({'nearest_furniture': 'stove'}, detected_objects + open_containers + ['daytime', 'nearest_furniture', 'positions']),
            ],
            "open": [
                ({'open(cupboard_door_left)': True}, detected_objects + open_containers + ['daytime', 'nearest_furniture', 'positions']),
                ({'open(fridge_door)': True}, detected_objects + open_containers + ['daytime', 'nearest_furniture', 'positions']),

            ],
            "apriori": [
                ({}, detected_objects + open_containers + ['daytime', 'nearest_furniture', 'positions'])
                # ({}, detected_objects)
            ],
        }

        for postype, queries in queries.items():
            print("POSTYPE:", postype)

            plotdir = os.path.join(locs.logs, f"Perception-{Path(self.recent_perception).stem}", f"test_reproduce_data_perception-{postype}")
            if not os.path.exists(os.path.join(locs.logs, f"Perception-{Path(self.recent_perception).stem}")):
                os.mkdir(os.path.join(locs.logs, f"Perception-{Path(self.recent_perception).stem}"))

            if not os.path.exists(plotdir):
                os.mkdir(plotdir)

            for i, (query, plots) in enumerate(queries):

                print("QUERY:", query)
                querystring = ';'.join([f'{vname}: {fmt(val, prec=1, positive=True)}' for vname, val in query.items()])
                prefix = f'Perception-' + '_'.join([f'{vname}_{fmt(val, prec=1, positive=True)}' for vname, val in query.items()])

                # print(len(j.allnodes), len(cond.allnodes))
                cond = j.conditional_jpt(
                    evidence=j.bind({k: v for k, v in query.items() if k in j.varnames},
                                    allow_singular_values=False
                                    ),
                    fail_on_unsatisfiability=False
                )

                post = j.posterior(
                    variables=[v for v in j.variables if v.name not in query],
                    evidence=j.bind({k: v for k, v in query.items() if k in j.varnames},
                        allow_singular_values=False
                    ),
                    fail_on_unsatisfiability=False
                )
                #
                # j.plot(
                #     filename=f'{prefix}-orig',
                #     directory=os.path.join(plotdir),
                #     leaffill='#CCDAFF',
                #     nodefill='#768ABE',
                #     alphabet=True,
                #     view=False
                # )
                # cond.plot(
                #     filename=f'{prefix}-conditional',
                #     directory=os.path.join(plotdir),
                #     leaffill='#CCDAFF',
                #     nodefill='#768ABE',
                #     alphabet=True,
                #     view=False
                # )

                for plot in plots:
                    print(f'Plotting {plot}')
                    limx = (0, 100)
                    limy = (0, 100)

                    # plot ground truth
                    if plot == "positions":

                        gt = go.Figure()
                        if addobstacles:

                            # plot obstacles in background
                            if self.obstacles is not None:
                                for (o, on) in self.obstacles:
                                    gt.add_trace(
                                        plotly_sq(o, lbl=on, color='rgb(59, 41, 106)', legend=False))

                        gt_ = plot_data_subset(
                            df,
                            xvar='x_in',
                            yvar='y_in',
                            limx=limx,
                            limy=limy,
                            constraints=query,
                            save=None,
                            show=False,
                            color='rgb(0,104,180)'
                        )
                        if addobstacles:
                            gt.layout = gt_.layout

                        gt.add_traces(gt_.data)

                        fig_to_file(gt, os.path.join(plotdir, f"{prefix}-{plot}-gt.html"), ftypes=['.svg', '.html'])

                        # data generation
                        x = np.linspace(*limx, 200)
                        y = np.linspace(*limy, 200)

                        X, Y = np.meshgrid(x, y)
                        Z = np.array([cond.pdf(cond.bind({'x_in': x, 'y_in': y})) for x, y, in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
                        lbl = np.full(Z.shape, '<br>'.join([f'{vname}: {val}' for vname, val in query.items()]))

                        data = pd.DataFrame(
                            data=[[x, y, Z, lbl]],
                            columns=['x', 'y', 'z', 'lbl']
                        )

                        hm = go.Figure()
                        if addobstacles:

                            # plot obstacles in background
                            if self.obstacles is not None:
                                for (o, on) in self.obstacles:
                                    hm.add_trace(
                                        plotly_sq(o, lbl=on, color='rgb(59, 41, 106)', legend=False))

                        # plot JPT Heatmap
                        hm_ = plot_heatmap(
                            xvar='x',
                            yvar='y',
                            data=data,
                            title=False,  # f'pdf({",".join([f"{vname}: {val}" for vname, val in pdfvars.items()])})',
                            limx=limx,
                            limy=limy,
                            show=False,
                            save=None,
                            showbuttons=False
                        )
                        if addobstacles:
                            hm.layout = hm_.layout

                        hm.add_traces(hm_.data)

                        fig_to_file(hm, os.path.join(plotdir, f"{prefix}-{plot}-dist-hm.html"), ftypes=['.svg', '.html'])

                        # plot JPT surface
                        sf = plot_heatmap(
                            xvar='x',
                            yvar='y',
                            data=data,
                            title=False,  # f'pdf({",".join([f"{vname}: {val}" for vname, val in pdfvars.items()])})',
                            limx=limx,
                            limy=limy,
                            show=False,
                            save=None,
                            showbuttons=False,
                            fun="surface"
                        )

                        fig_to_file(sf, os.path.join(plotdir, f"{prefix}-{plot}-dist-3d.html"), ftypes=['.svg', '.html'])
                    else:
                        if plot in post:
                            gt = plot_data_subset(
                                df,
                                xvar=plot,
                                yvar=None,
                                constraints=query,
                                save=None,
                                show=False,
                                plot_type="histogram",
                                normalize=True,
                                color='rgb(0,104,180)'
                            )
                            fig_to_file(gt, os.path.join(plotdir, f"{prefix}-{plot}-gt.html"), ftypes=['.svg', '.html'])

                            # plot distribution of variable
                            print('PLOTTING DIST', plot)
                            dist = post[plot].plot(
                                view=False,
                                title=False,  # f'Dist: {plot}<br>Query: {querystring}',
                                alphabet = True,
                                color='rgb(59, 41, 106)'
                            )
                            fig_to_file(dist, os.path.join(plotdir, f"{prefix}-{plot}-dist.html"), ftypes=['.svg', '.html'])
                            

    def test_reproduce_data_pr2(self) -> None:
        # MULTIPLE sets of constraints:
        # for any constrained PR2 variables, plot all remaining dists
        import plotly.express as px

        # load data and JPT that has been learnt from this data
        logger.info(f"Using tree from {self.recent_pr2}")
        j = self.models['000-pr2.tree']
        df = pd.read_parquet(os.path.join(self.recent_pr2, 'data', f'000-pr2.parquet'))

        # constraints/query values
        # the postype determines a category, tp
        queries_ = {
            "bodyparts": [
                ({"bodyPartsUsed": ":LEFT"}, ["type", "object_acted_on", "success"]),
                ({"bodyPartsUsed": ":RIGHT"}, ["type", "object_acted_on", "success"]),
            ],
            "failure": [  # failed actions
                ({'success': False}, ['type', "failure", "positions"]),
                ({'type': "Grasping", "success": False}, ["failure", "positions"]),
                ({'type': "Placing", "success": False}, ["failure", "positions"]),
            ],
            "success": [
                ({"success": True}, ['type', "failure", "positions"]),
                ({"success": True, 'type': "Grasping"}, ["bodyPartsUsed", "positions"]),
                ({"success": True, 'type': "Placing"}, ["bodyPartsUsed", "positions"]),
                ({"success": True, 'object_acted_on': 'milk_1'}, ["type"]),
            ],
            "apriori": [
                ({}, ['type', "arm", "bodyPartsUsed", "success", "object_acted_on", "failure", "positions"]),
            ],
        }

        for postype, queries in queries_.items():
            logger.info("POSTYPE:", postype)

            plotdir = os.path.join(locs.logs, f"PR2-{Path(self.recent_pr2).stem}", f"test_reproduce_data_pr2-{postype}")
            if not os.path.exists(os.path.join(locs.logs, f"PR2-{Path(self.recent_pr2).stem}")):
                os.mkdir(os.path.join(locs.logs, f"PR2-{Path(self.recent_pr2).stem}"))
            if not os.path.exists(plotdir):
                os.mkdir(plotdir)

            for i, (query, plots) in enumerate(queries):

                logger.info("QUERY:", query)
                querystring = ';'.join([f'{vname}: {fmt(val, prec=1, positive=True)}' for vname, val in query.items()])
                prefix = f'PR2-' + '_'.join([f'{vname}_{fmt(val, prec=1, positive=True)}' for vname, val in query.items()])

                # generate tree conditioned on given position and/or direction
                cond = j.conditional_jpt(
                    evidence=j.bind(
                        {k: v for k, v in query.items() if k in j.varnames},
                        allow_singular_values=False
                    ),
                    fail_on_unsatisfiability=False
                )

                # calculate posterior from query
                post = j.posterior(
                    variables=[v for v in j.variables if v.name not in query],
                    evidence=j.bind({k: v for k, v in query.items() if k in j.varnames},
                        allow_singular_values=False
                    ),
                    fail_on_unsatisfiability=False
                )

                # j.plot(
                #     filename=f'{prefix}-orig',
                #     directory=os.path.join(plotdir),
                #     leaffill='#CCDAFF',
                #     nodefill='#768ABE',
                #     alphabet=True,
                #     view=False
                # )
                # cond.plot(
                #     filename=f'{prefix}-conditional',
                #     directory=os.path.join(plotdir),
                #     leaffill='#CCDAFF',
                #     nodefill='#768ABE',
                #     alphabet=True,
                #     view=False
                # )

                for plot in plots:
                    logger.info(f'Plotting {plot}')

                    # plot ground truth
                    if plot == "positions":
                        gt = plot_data_subset(
                            df,
                            xvar='t_x',
                            yvar='t_y',
                            constraints=query,
                            save=None,
                            show=False,
                            color='rgb(0,104,180)'
                        )
                        fig_to_file(gt, os.path.join(plotdir, f"{prefix}-{plot}-gt.html"), ftypes=['.svg', '.html'])

                        # data generation
                        limx = (-3, 1)
                        limy = (-1.5, 0.3)
                        x = np.linspace(*limx, 200)
                        y = np.linspace(*limy, 200)

                        X, Y = np.meshgrid(x, y)
                        Z = np.array([cond.pdf(cond.bind({'t_x': x, 't_y': y})) for x, y, in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
                        lbl = np.full(Z.shape, '<br>'.join([f'{vname}: {val}' for vname, val in query.items()]))

                        data = pd.DataFrame(
                            data=[[x, y, Z, lbl]],
                            columns=['x', 'y', 'z', 'lbl']
                        )

                        # plot JPT Heatmap
                        dist = plot_heatmap(
                            xvar='x',
                            yvar='y',
                            data=data,
                            title=False,  # f'pdf({",".join([f"{vname}: {val}" for vname, val in pdfvars.items()])})',
                            limx=limx,
                            limy=limy,
                            show=False,
                            save=None,
                            showbuttons=False
                        )
                        fig_to_file(dist, os.path.join(plotdir, f"{prefix}-{plot}-dist-hm.html"), ftypes=['.svg', '.html'])

                        # plot JPT Heatmap
                        dist = plot_heatmap(
                            xvar='x',
                            yvar='y',
                            data=data,
                            title=False,  # f'pdf({",".join([f"{vname}: {val}" for vname, val in pdfvars.items()])})',
                            limx=limx,
                            limy=limy,
                            show=False,
                            save=None,
                            showbuttons=False,
                            fun='surface'
                        )
                        fig_to_file(dist, os.path.join(plotdir, f"{prefix}-{plot}-dist-3d.html"), ftypes=['.svg', '.html'])
                    else:
                        if plot in post:
                            gt = plot_data_subset(
                                df,
                                xvar=plot,
                                yvar=None,
                                save=None,
                                show=False,
                                constraints=query,
                                plot_type="histogram",
                                normalize=True,
                                color='rgb(0,104,180)',
                            )
                            fig_to_file(gt, os.path.join(plotdir, f"{prefix}-{plot}-gt.html"), ftypes=['.svg', '.html'])

                            # plot distribution of variable
                            dist = post[plot].plot(
                                view=False,
                                title=False,
                                alphabet=True,
                                color='rgb(59, 41, 106)'
                            )
                            fig_to_file(dist, os.path.join(plotdir, f"{prefix}-{plot}-dist.html"), ftypes=['.svg', '.html'])

    def test_jpt_leaves_plot(self):
        j = self.models['000-move.tree']
        plot_tree_leaves(j, j.varnames['x_in'], j.varnames['y_in'], limx=(0, 100), limy=(0, 100), show=True)

    def test_jpt_prune(self):
        # j = self.models['000-move.tree']
        f = os.path.join(recent_example(os.path.join(locs.examples, 'move_exp')), '000-move_exp.tree')
        j = JPT.load(f)
        j.postprocess_leaves()
        j_ = j.prune(.7)
        j_.save(f)

    def test_astar_cram_path(self) -> None:
        initx, inity, initdirx, initdiry = [20, 70, 0, -1]
        shift = True
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

        initstate = State()
        initstate.update(
            {
                'x_in': distx,
                'y_in': disty,
                'xdir_in': distdx,
                'ydir_in': distdy
            }
        )

        # alter this and comment out cmds below for playing around
        cmds = [
            {'tree': '000-move.tree', 'params': {'action': 'move'}},
            {'tree': '000-move.tree', 'params': {'action': 'move'}},
            {'tree': '000-turn.tree', 'params': {'action': 'turn', 'angle': -15}},
            {'tree': '000-turn.tree', 'params': {'action': 'turn', 'angle': -5}},
            {'tree': '000-turn.tree', 'params': {'action': 'turn', 'angle': -15}},
            {'tree': '000-move.tree', 'params': {'action': 'move'}},
            {'tree': '000-turn.tree', 'params': {'action': 'turn', 'angle': -10}},
            {'tree': '000-move.tree', 'params': {'action': 'move'}},
            {'tree': '000-turn.tree', 'params': {'action': 'turn', 'angle': 15}},
            {'tree': '000-move.tree', 'params': {'action': 'move'}},
            {'tree': '000-move.tree', 'params': {'action': 'move'}},
            {'tree': '000-turn.tree', 'params': {'action': 'turn', 'angle': 30}},
        ]

        # do not touch, diss-plot configuration!
        cmds = [
            {'tree': '000-turn.tree', 'params': {'action': 'turn', 'angle': -15}},
            {'tree': '000-move.tree', 'params': {'action': 'move'}},
            {'tree': '000-move.tree', 'params': {'action': 'move'}},
            {'tree': '000-turn.tree', 'params': {'action': 'turn', 'angle': -15}},
            {'tree': '000-move.tree', 'params': {'action': 'move'}},
            {'tree': '000-turn.tree', 'params': {'action': 'turn', 'angle': -10}},
            {'tree': '000-move.tree', 'params': {'action': 'move'}},
            {'tree': '000-turn.tree', 'params': {'action': 'turn', 'angle': 15}},
            {'tree': '000-move.tree', 'params': {'action': 'move'}},
            {'tree': '000-move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-turn.tree', 'params': {'action': 'turn', 'angle': -12}},
            # {'tree': '000-move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-turn.tree', 'params': {'action': 'turn', 'angle': -5}},
            # {'tree': '000-move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-turn.tree', 'params': {'action': 'turn', 'angle': 3}},
            # {'tree': '000-move.tree', 'params': {'action': 'move'}},
            # {'tree': '000-turn.tree', 'params': {'action': 'turn', 'angle': 15}},
        ]

        # VARIANT II: each leaf of the conditional tree represents one possible action
        s = initstate
        p = [[s, {}]]
        for i, cmd in enumerate(cmds):
            print(f'Step {i} of {len(cmds)}: {cmd["params"]["action"]}({cmd["params"].get("angle", "")})')
            t = self.models[cmd['tree']]

            # generate evidence by using intervals from the 5th percentile to the 95th percentile for each distribution
            evidence = {
                var: ContinuousSet(s[var].ppf(.05), s[var].ppf(.95)) for var in s.keys() if var != 'collided'
            }

            if cmd["params"] is not None:
                evidence.update(cmd["params"])

            # candidate is the conditional tree
            # t_ = self.generate_steps(evidence, t)
            # best = t.posterior(
            #     variables=t.targets,
            #     evidence=t.bind({k: v for k, v in evidence.items() if k in t.varnames},
            #         allow_singular_values=False
            #     ),
            #     fail_on_unsatisfiability=False
            # )
            cond = t.conditional_jpt(
                evidence=t.bind({k: v for k, v in evidence.items() if k in t.varnames},
                    allow_singular_values=False
                ),
                fail_on_unsatisfiability=False
            )

            best = cond.posterior(variables=t.targets,)

            if best is None:
                print('skipping command', cmd, 'unsatisfiable!')
                continue

            # create successor state
            s_ = State()
            s_.update({k: v for k, v in s.items()})
            s_.tree = cmd['tree']
            s_.leaf = None

            # update belief state of potential predecessor
            for vn, d in best.items():
                vname = vn.name
                outvar = vn.name.replace('_in', '_out')
                invar = vn.name.replace('_out', '_in')

                if vname.endswith('_out') and vname.replace('_out', '_in') in s_:
                    # if the _in variable is already contained in the state, update it by adding the delta
                    # from the leaf distribution
                    if shift:
                        s_[invar] = Numeric().set(
                            QuantileDistribution.from_cdf(s_[invar].cdf.xshift(-best[outvar].expectation())))
                    else:
                        indist = s_[invar]
                        outdist = best[outvar]
                        if len(indist.cdf.functions) > 20:
                            print(f"A Approximating {invar} distribution of s_ with {len(indist.cdf.functions)} functions")
                            indist = indist.approximate(n_segments=20)
                            # s_[invar] = s_[invar].approximate_fast(eps=.01)
                        if len(outdist.cdf.functions) > 20:
                            print(f"B Approximating {outvar} distribution of best with {len(outdist.cdf.functions)} functions")
                            outdist = outdist.approximate(n_segments=20)
                            # best[outvar] = best[outvar].approximate_fast(eps=.01)
                        vname = invar
                        s_[vname] = indist + outdist
                elif vname.endswith('_in') and vname in s_:
                    # do not overwrite '_in' distributions
                    continue
                else:
                    s_[vname] = d

                if not shift:
                    if hasattr(s_[vname], 'approximate'):
                        print(f"C Approximating {vname} distribution of s_ (result) with {len(s_[vname].cdf.functions)} functions")
                        s_[vname] = s_[vname].approximate(n_segments=20)

            p.append([s_, cmd['params']])
            s = State()
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
            save=os.path.join(locs.logs, f'test_astar_cram_path.svg'),
            obstacles=obstacles,
            show=False
        )

        fig.write_html(
            os.path.join(locs.logs, f'test_astar_cram_path.html'),
            config=defaultconfig(os.path.join(locs.logs, f'test_astar_cram_path.html')),
            include_plotlyjs="cdn"
        )

        fig.show(config=defaultconfig(os.path.join(locs.logs, f'test_astar_cram_path.html')))

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

        # plot animation of collision bar chart representing change of collision status
        # frames = [s['collided'].plot(view=False).data for (s, _) in p if 'collided' in s]
        # plotly_animation(
        #     data=frames,
        #     save=os.path.join(locs.logs, f'collision.html'),
        #     show=True
        # )

        plot_dir(
            path=p,
            save=os.path.join(locs.logs, f'test_astar_cram_path-dirxy.html'),
            show=True,
            limx=(-3, 3),
            limy=(-3, 3)
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

    def test_two_gaussians_diff(self) -> None:

        g1x = Gaussian(-.25, .2)
        g1y = Gaussian(-.25, .1)
        d1x = g1x.sample(500)
        d1y = g1y.sample(500)
        dist1x = Numeric()
        dist1y = Numeric()
        dist1x.fit(d1x.reshape(-1, 1), col=0)
        dist1y.fit(d1y.reshape(-1, 1), col=0)

        g2x = Gaussian(1, .2)
        g2y = Gaussian(.5, .05)
        d2x = g2x.sample(500)
        d2y = g2y.sample(500)
        dist2x = Numeric()
        dist2y = Numeric()
        dist2x.fit(d2x.reshape(-1, 1), col=0)
        dist2y.fit(d2y.reshape(-1, 1), col=0)

        dist3x = Numeric().set(QuantileDistribution.from_cdf(dist2x.cdf.xshift(-5)))
        dist3y = Numeric().set(QuantileDistribution.from_cdf(dist2y.cdf.xshift(-1)))

        plot_multiple_dists(
            [[dist1x, dist1y], [dist2x, dist2y], [dist3x, dist3y]],
            # [[dist2x, dist2y], [dist3x, dist3y]],
            limx=(-2, 8),
            limy=(2, 2),
            show=True
        )

    def test_move_till_collision(self) -> None:
        # position near obstacle or wall and move a couple of stepps, observe state of collision variable
        print("loading example", self.recent_move)

        initx, inity, initdirx, initdiry = [7, 70, -1, 0]
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

        initstate = State()
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
        t = self.models['000-move.tree']
        for i, step in enumerate(range(4)):
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

            # cond = t.conditional_jpt(
            #     evidence=t.bind({k: v for k, v in evidence.items() if k in t.varnames},
            #         allow_singular_values=False
            #     ),
            #     fail_on_unsatisfiability=False
            # )
            # best = cond.posterior(variables=t.targets)

            if best is None:
                print('skipping at step', step, 'unsatisfiable!')
                continue

            # create successor state
            s_ = State()
            s_.update({k: v for k, v in s.items()})
            s_.tree = '000-move.tree'
            s_.leaf = None

            # update belief state of potential predecessor
            print("Updating new state...")
            for vn, d in best.items():
                outvar = vn.name
                invar = vn.name.replace('_out', '_in')
                if outvar != invar and invar in s_:
                    # if the _in variable is already contained in the state, update it by adding the delta
                    # from the leaf distribution
                    try:
                        if len(s_[invar].cdf.functions) > 20:
                            s_[invar] = s_[invar].approximate(n_segments=20)
                        if len(best[outvar].cdf.functions) > 20:
                            best[outvar] = best[outvar].approximate(n_segments=20)

                        s_[invar] = s_[invar] + best[outvar]
                    except:
                        print(f"Crashing belief state for variable\n{outvar}:\n{best[outvar].pdf}\n\n{best[outvar].cdf}")
                        traceback.print_exc()
                else:
                    s_[invar] = d

                if hasattr(s_[invar], 'approximate'):
                    s_[invar] = s_[invar].approximate(n_segments=20)

            p.append([s_, {'action': 'move'}])
            s = State()
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
            save=os.path.join(locs.logs, f'test_move_till_collision.svg'),
            obstacles=obstacles,
            show=False
        )

        fig.write_html(
            os.path.join(locs.logs, f'test_move_till_collision.html'),
            config=defaultconfig(os.path.join(locs.logs, f'test_move_till_collision.html')),
            include_plotlyjs="cdn"
        )

        fig.show(config=defaultconfig(os.path.join(locs.logs, f'test_move_till_collision.html')))

        # print heatmap representing position distribution update
        plot_pos(
            path=p,
            save=os.path.join(locs.logs, f'test_move_till_collision-pos-animation.html'),
            show=True,
            limx=(0, 100),
            limy=(0, 100)
        )

        # plot animation of collision bar chart representing change of collision status
        frames = [s['collided'].plot(view=False).data for (s, _) in p if 'collided' in s]
        plotly_animation(
            data=frames,
            save=os.path.join(locs.logs, f'test_move_till_collision-collision-anmiation.html'),
            show=True
        )

    def test_reduce_tree(self):
        t = self.models['000-move.tree']

        for l in t.leaves.values():
            for var, d in l.distributions.items():
                if hasattr(d, 'approximate') and len(d.cdf.functions) > 20:
                    print(f'Approximating dist {var.name} of leaf {l.idx} with {len(d.cdf.functions)} functions')
                    l.distributions[var] = d.approximate(n_segments=20)

        t.save(os.path.join(self.recent_move, '000-move-approximated.tree'))

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
            os.path.join(locs.logs, 'test_plot_kaleido_error.png'),
            scale=1
        )

    def test_data_point_star(self) -> None:
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
                turn(*(0,1), 20*i) for i in range(18)
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
            save=os.path.join(locs.logs, 'test_data_point_star.svg'),
        )

        fig.add_traces(fig_.data)
        fig.layout = fig_.layout
        fig.update_layout(
            height=1000,
            width=1000,
            xaxis=dict(
                title='x',
                range=(-1, 1)
            ),
            yaxis=dict(
                title='y',
                range=(-1, 1)
            )
        )

        fig.show(config=defaultconfig())

    def test_data_point_one_turn(self) -> None:
        # plot image representing 3 single turns
        def turn(x, y, deg):
            deg = np.radians(-deg)
            return x * math.cos(deg) - y * math.sin(deg), x * math.sin(deg) + y * math.cos(deg)

        d = [
            (
                0,
                0,
                *(1, 0),
                f"Step {0}",
                "",
                1
            ),
            (
                0,
                0,
                *turn(*(1, 0), -50),
                f"Step {1}",
                "",
                1
            ),
            (
                0,
                0,
                *turn(*(1, 0), -45),
                f"Step {1}",
                "",
                1
            ),
            (
                0,
                0,
                *turn(*(1, 0), -40),
                f"Step {1}",
                "",
                1
            )
        ]

        data = pd.DataFrame(
            data=d,
            columns=['x', 'y', 'dx', 'dy', 'step', 'lbl', 'size']
        )

        fig = go.Figure()
        fig_ = plot_scatter_quiver(
            'x',
            'y',
            data,
            title=None,
            save=os.path.join(locs.logs, 'test_data_point_one_turn.svg'),
        )

        fig.add_traces(fig_.data)
        fig.layout = fig_.layout
        fig.update_layout(
            height=1000,
            width=1000,
            xaxis=dict(
                title='x',
                range=(-.5, .5)
            ),
            yaxis=dict(
                title='y',
                range=(-.5, .5)
            )
        )

        fig.show(config=defaultconfig())

    def test_data_point_one_move(self) -> None:

        d = [
            (
                0,
                0,
                *(1, 0),
                f"Step {0}",
                "",
                1
            ),
            (
                1,
                0,
                *(1, 0),
                f"Step {1}",
                "",
                1
            )
        ]

        data = pd.DataFrame(
            data=d,
            columns=['x', 'y', 'dx', 'dy', 'step', 'lbl', 'size']
        )

        fig = go.Figure()
        fig_ = plot_scatter_quiver(
            'x',
            'y',
            data,
            title=None,
            save=os.path.join(locs.logs, 'test_data_point_one_move.svg'),
        )

        fig.add_traces(fig_.data)
        fig.layout = fig_.layout
        fig.update_layout(
            height=1000,
            width=1000,
            xaxis=dict(
                title='x',
                range=(-.5, 2)
            ),
            yaxis=dict(
                title='y',
                range=(-.5, .5)
            )
        )

        fig.show(config=defaultconfig())

    def tearDown(self) -> None:
        # draw path steps into grid (use action symbols)
        print()

    @classmethod
    def tearDownClass(cls) -> None:
        print()
