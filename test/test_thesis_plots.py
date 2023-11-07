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

from calo.utils import locs
from calo.utils.plotlib import plot_heatmap, plot_data_subset
from calo.utils.utils import recent_example
from jpt import SymbolicType, NumericVariable, JPT
from jpt.base.intervals import ContinuousSet
from jpt.distributions import Gaussian, Numeric

def generate_gaussian_samples(gaussians, n):
    per_gaussian = int(n / len(gaussians))
    data = [g.sample(per_gaussian) for g in gaussians]
    colors = [[c] * per_gaussian for c in ['#ff0000', '#0000ff'][:len(gaussians)]]

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

       cls.recent = recent_example(os.path.join(locs.examples, 'robotaction'))
       cls.recent = os.path.join(locs.examples, 'robotaction', '2023-11-02_14:50')

       cls.models = dict(
           [
               (
                   treefile.name,
                   JPT.load(str(treefile))
               )
               for p in [cls.recent]
               for treefile in Path(p).rglob('*.tree')
           ]
       )

    def test_plot_dist_similarity_discrete(self) -> None:
        # Arrange
        DistA = SymbolicType('DistA', labels=['A', 'B', 'C'])
        DistB = SymbolicType('DistB', labels=['D', 'E', 'F'])
        d1 = DistA().set(params=[.5, .25, .25])
        d2 = DistB().set(params=[.5, .25, .25])
        d3 = DistA().set(params=[.2, .1, .7])

        # Act
        mainfig = go.Figure()

        d1fig = d1.plot(
            title="$\\text{dist}_1$",
            color="0,104,180",
            view=False,
            horizontal=False
        )

        d2fig = d2.plot(
            title="$\\text{dist}_2$",
            color="134, 129, 177",
            view=False,
            horizontal=False
        )

        d3fig = d3.plot(
            title="$\\text{dist}_3$",
            color="138, 203, 183",
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
                range=None #  ['A', 'B', 'C', 'D', 'E', 'F']
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

        dx = Gaussian(mu1, v1).sample(50)
        distx = Numeric()
        distx.fit(dx.reshape(-1, 1), col=0)

        dxdelta = Gaussian(mu2, v2).sample(50)
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

    def test_rterror_conditional_jpt(self) -> None:
        # load data and JPT that has been learnt from this data
        j: JPT = ThesisPlotsTests.models['000-MOVEFORWARD.tree']
        print(f"Loading tree from {ThesisPlotsTests.recent}")

        # minimal set of preconditions that cause runtime error
        xdir_in = 0
        ydir_in = 1

        # This will create the error "RuntimeError: This should never happen. JPT.conditional_jpt() seems to be
        # broken :(" in jpt-dev/src/jpt/trees.py:2216
        pdfvars = {
            'xdir_in': xdir_in,
            'ydir_in': ydir_in,
            'collided': True
        }

        # generate tree conditioned on given position and/or direction
        cond = j.conditional_jpt(
            evidence=j.bind(
                {k: v for k, v in pdfvars.items() if k in j.varnames},
                # allow_singular_values=False
            ),
            fail_on_unsatisfiability=True
        )
        cond.plot(plotvars=['x_in', 'y_in', 'x_out', 'y_out', 'xdir_in', 'ydir_in', 'collided'], view=True)

    def test_reproduce_data_find_limits(self) -> None:
        # -> constrain target pos, plot feature pos

        # load data and JPT that has been learnt from this data
        j = ThesisPlotsTests.models['000-MOVEFORWARD.tree']
        print(f"Loading tree from {ThesisPlotsTests.recent}")

        # set settings
        limx = (-100, 100)
        limy = (-100, 100)

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
        x = np.linspace(*limx, 400)
        y = np.linspace(*limy, 400)

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

        # plot JPT
        plot_heatmap(
            xvar='x',
            yvar='y',
            data=data,
            title=f'pdf({",".join([f"{vname}: {val}" for vname, val in pdfvars.items()])})',
            limx=limx,
            limy=limy,
            show=True,
        )

        # plot ground truth
        df = pd.read_csv(
            os.path.join(ThesisPlotsTests.recent, 'data', f'000-ALL-MOVEFORWARD.csv'),
            delimiter=',',
            header=0
        )
        plot_data_subset(
            df,
            xvar="x_in",
            yvar="y_in",
            constraints=pdfvars,
            limx=limx,
            limy=limy,
            show=True
        )

    def test_reproduce_data_single_jpt(self) -> None:
        # -> constrain feature pos, plot target pos

        # load data and JPT that has been learnt from this data
        j = ThesisPlotsTests.models['000-MOVEFORWARD.tree']
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
            show=True,
        )

        # plot ground truth
        df = pd.read_csv(
            os.path.join(ThesisPlotsTests.recent, 'data', f'000-ALL-MOVEFORWARD.csv'),
            delimiter=',',
            header=0
        )
        plot_data_subset(
            df,
            "x_out",
            "y_out",
            pdfvars,
            limx=limx,
            limy=limy,
            show=True
        )

    def test_reproduce_data_multiple_jpt(self) -> None:
        # load data and JPT that has been learnt from this data
        j = ThesisPlotsTests.models['000-MOVEFORWARD.tree']

        # set settings
        limx = (-3, 3)
        limy = (-3, 3)

        v = ['xdir_in', 'ydir_in', None]
        dirs = [-1, .5, 0, .5, 1]
        psx = [-75, -50, -25, -15, -10, 0, 20, 25, 50]
        psy = [-75, -50, -40, -30, -10, 10, 25, 40, 50]

        for y_ in psy:
            for x_ in psx:
                i = 0
                for v_ in v:
                    for xd in dirs:
                        i += 1
                        locs.logs

                        plotdir = os.path.join(locs.logs, f"pos{x_:+d}{y_:+d}")
                        if not os.path.exists(plotdir):
                            os.mkdir(plotdir)

                        x_in = x_
                        y_in = y_
                        xdir_in = xd
                        ydir_in = xd

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

                        pdfvars = {
                            'x_in': ContinuousSet(xmin, xmax),
                            'y_in': ContinuousSet(ymin, ymax),
                            # 'xdir_in': ContinuousSet(xdirmin, xdirmax),
                            # 'ydir_in': ContinuousSet(ydirmin, ydirmax),
                        }

                        if v_ is not None:
                            vmin = xd - tolerance
                            vmax = xd + tolerance
                            pdfvars[v_] = ContinuousSet(vmin, vmax)

                        # constraints is a list of 3-tuples: ('<column name>', 'operator', value)
                        constraints = [(var, op, v) for var, val in pdfvars.items() for v, op in [(val.lower, ">="), (val.upper, "<=")]]
                        print('\nConstraints on dataset: ', constraints)

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
                            show=False,
                            save=os.path.join(plotdir, f"{i}-{v_}{xd:+.0f}.svg")
                        )

                        # plot ground truth
                        df = pd.read_csv(
                            os.path.join(ThesisPlotsTests.recent, 'data', f'000-ALL-MOVEFORWARD.csv'),
                            delimiter=',',
                            header=0
                        )
                        plot_data_subset(
                            df,
                            constraints,
                            limx=limx,
                            limy=limy,
                            save=os.path.join(plotdir, f"{i}-{v_}{xd:+.0f}-gt.svg"),
                            show=False
                        )

    def tearDown(self) -> None:
        # draw path steps into grid (use action symbols)
        print()

    @classmethod
    def tearDownClass(cls) -> None:
        print()
