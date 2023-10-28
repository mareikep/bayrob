import os
import random
from typing import List, Tuple, Dict, Any

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from _plotly_utils.colors import sample_colorscale
from plotly.graph_objs import Figure

import numpy as np
import pandas as pd
from calo.utils import locs
from calo.utils.utils import unit
from dnutils import first, ifnone

from jpt import JPT
from jpt.distributions import Gaussian

defaultconfig = dict(
    displaylogo=False,
    toImageButtonOptions=dict(
        format='svg',  # one of png, svg, jpeg, webp
        filename='calo_plot',
        height=500,
        width=700,
        scale=1  # Multiply title/legend/axis/canvas sizes by this factor
    ),
   #  modeBarButtonsToAdd=[  # allow drawing tools to highlight important regions before downloading snapshot
   #      'drawline',
   #      'drawopenpath',
   #      'drawclosedpath',
   #      'drawcircle',
   #      'drawrect',
   #      'eraseshape'
   # ]
)

def plot_pos(
        path: List,
        title: str = None,
        conf: float = None,
        limx: Tuple = None,
        limy: Tuple = None,
        limz: Tuple = None,
        save: str = None,
        show: bool = True
) -> Figure:

    # if no title is given, generate it according to the input
    if title is None:
        title = f'Position x/y'

    # generate datapoints
    data = pd.DataFrame(
            data=[
                gendata(
                    'x_in',
                    'y_in',
                    s,
                    p,
                    conf=conf
                ) for i, (s, p) in enumerate(path)
            ],
            columns=['x', 'y', 'z', 'lbl']
        )

    # generate datapoints
    # data = [(gendata_out if inout else gendata)('x_in', 'y_in', s, p, conf=conf) for s, p in path]
    # data = [
    #     gendata(
    #         'x_in',
    #         'y_in',
    #         s,
    #         p,
    #         conf=conf
    #     ) for s, p in path
    # ]

    return plot_heatmap(
        xvar='x',
        yvar='y',
        data=data,
        title=title,
        limx=limx,
        limy=limy,
        limz=limz,
        save=save,
        show=show
    )


def plot_dir(
        path: List,
        title: str = None,
        conf: float = None,
        limx: Tuple = None,
        limy: Tuple = None,
        limz: Tuple = None,
        save: str = None,
        show: bool = True
) -> Figure:

    if title is None:
        title = f'Direction xdir/ydir'

    # generate datapoints
    data = [
        pd.DataFrame(
            data=[
                gendata(
                    'xdir_in',
                    'ydir_in',
                    s,
                    p,
                    conf=conf
                )
            ],
            columns=['xdir_in', 'ydir_in', 'z', 'lbl']
        ) for s, p in path
    ]

    # # generate datapoints
    # data = [
    #     gendata(
    #         'xdir_in',
    #         'ydir_in',
    #         s,
    #         p,
    #         conf=conf
    #     ) for s, p in path
    # ]

    return plot_heatmap(
        xvar='xdir',
        yvar='ydir',
        data=data,
        title=title,
        limx=limx,
        limy=limy,
        limz=limz,
        save=save,
        show=show
    )


def gendata(
        xvar,
        yvar,
        state,
        params: Dict = None,
        conf: float = None
):

    # generate datapoints
    x = state[xvar].pdf.boundaries()
    y = state[yvar].pdf.boundaries()

    X, Y = np.meshgrid(x, y)
    Z = np.array(
        [
            state[xvar].pdf(x) * state[yvar].pdf(y)
            for x, y, in zip(X.ravel(), Y.ravel())
        ]).reshape(X.shape)

    # show only values above a certain threshold, consider lower values as high-uncertainty areas
    if conf is not None:
        Z[Z < conf] = 0.

    # remove or replace by eliminating values > median
    Z[Z > np.median(Z)] = np.median(Z)

    lbl = f'Leaf#{state.leaf if state.leaf is not None else "ROOT"} ' \
          f'Action: {params.get("action")}, Params: {"angle=" if "angle" in params else ""}{params.get("angle", None)}'

    return x, y, Z, lbl


def plot_heatmap(
        xvar: str,
        yvar: str,
        data: pd.DataFrame,
        title: str = None,
        limx: Tuple = None,
        limy: Tuple = None,
        limz: Tuple = None,
        save: str = None,
        show: bool = True,
) -> Figure:
    """Plot heatmap or 3D surface plot with plotly
    """

    # determine limits, if not given
    # exclude first element from limit calculations, as it causes the lower probs to be all equal colors
    if limx is None:
        limx = min([df[xvar].min() for _, df in data.iterrows()]), max([df[xvar].max() for _, df in data.iterrows()])

    if limy is None:
        limy = min([df[yvar].min() for _, df in data.iterrows()]), max([df[yvar].max() for _, df in data.iterrows()])

    if limz is None:
        limz = min([df["z"].min() for _, df in data.iterrows()]), max([df["z"].max() for _, df in data.iterrows()])

    # generate the frames
    frames = [
        go.Frame(
            data=go.Heatmap(
                x=d[xvar],  # x[0],
                y=d[yvar].T,  # y.T[0],
                z=d['z'],  # z,
                zmin=limz[0],
                zmax=limz[1],
                customdata=np.full(d['z'].shape, d["lbl"]),
                colorscale=px.colors.sequential.dense,
                colorbar=dict(
                    title=f"P({xvar},{yvar})",
                    orientation='h',
                    titleside="top",
                    x=.5,
                    y=-.3
                ),
                hovertemplate='x: %{x}<br>'
                              'y: %{y}<br>'
                              'z: %{z}'
                              '<extra>%{customdata}</extra>'
            ),
            name=f"Step {i}"
        ) for i, d in data.iterrows()
    ]

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
    )

    def frame_args(duration):
        return dict(
            frame=dict(
                duration=duration,
                redraw=True
            ),
            mode="immediate",
            fromcurrent=True,
            transition=dict(
                duration=0,
                redraw=True
            )
        )

    steps = []
    for i, f in enumerate(fig.frames):
        step = dict(
            args=[
                [f.name],
                frame_args(0)
            ],
            label=f.name,
            method="animate"
        )
        steps.append(step)

    sliders = [
        dict(
            currentvalue=dict(
                prefix="Step: "
            ),
            steps=steps
        )
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                x=-.05,
                y=-.225,
                direction="right",
                yanchor="top",
                buttons=list(
                    [
                        dict(
                            args=[
                                None,
                                frame_args(100)
                            ],
                            label=f'{"&#9654;":{" "}{"^"}{20}}',
                            method="animate"
                        ),
                        dict(
                            args=[
                                [
                                    None
                                ],
                                dict(
                                    frame=dict(
                                        duration=0,
                                        redraw=False
                                    ),
                                    mode="immediate",
                                    transition=dict(
                                        duration=0
                                    )
                                )
                            ],
                            label=f'{"&#9208;":{" "}{"^"}{20}}',
                            method="animate"
                        ),
                    ]
                )
            ),
            dict(
                type="buttons",
                x=-.05,
                y=-.3,
                direction="right",
                yanchor="top",
                buttons=list([
                    dict(
                        args=["type", "surface"],
                        label=f'{"3D Surface":{" "}{"^"}{15}}',
                        method="restyle"
                    ),
                    dict(
                        args=["type", "heatmap"],
                        label=f'{"Heatmap":{" "}{"^"}{15}}',
                        method="restyle"
                    )
                ])
            )
        ],
        sliders=sliders,
        height=1000,
        width=1000,
        xaxis=dict(
            title=xvar,
            tickangle=-45,
            side='top',
            range=[*limx]
        ),
        yaxis=dict(
            title=yvar,
            range=[*limy]
        ),
        title=title
    )

    if save:
        if save.endswith('html'):
            fig.write_html(
                save,
                config=defaultconfig,
                include_plotlyjs="cdn"
            )
        else:
            fig.write_image(save)

    if show:
        fig.show(config=defaultconfig)

    return fig


def plot_tree_dist(
    tree: JPT,
    qvarx: Any = None,
    qvary: Any = None,
    title: str = None,
    conf: float = None,
    limx: Tuple = None,
    limy: Tuple = None,
    limz: Tuple = None,
    save: str = None,
    show: bool = True
) -> Figure:
    """Plots a heatmap representing the belief state for the agents' position, i.e. the joint
    probability of the x and y variables: P(x, y).

    :param title: The plot title
    :param conf:  A confidence value. Values below this threshold are set to 0. (= equal color for lowest value in plot)
    :param limx: The limits for the x-variable; determined from boundaries if not given
    :param limy: The limits for the y-variable; determined from boundaries if not given
    :param limz: The limits for the z-variable; determined from data if not given
    :param save: The location where the plot is saved (if given)
    :param show: Whether the plot is shown
    :return: None
    """

    # generate datapoints
    x = np.linspace(limx[0], limx[1], 50)
    y = np.linspace(limy[0], limy[1], 50)

    X, Y = np.meshgrid(x, y)
    Z = np.array(
        [
            tree.pdf(
                tree.bind(
                    {
                        qvarx: x,
                        qvary: y
                    }
                )
            ) for x, y, in zip(X.ravel(), Y.ravel())
        ]

    ).reshape(X.shape)

    lbl = f'Some random label'

    # show only values above a certain threshold, consider lower values as high-uncertainty areas
    if conf is not None:
        Z[Z < conf] = 0.

    data = pd.DataFrame(data=[[x, y, Z, lbl]], columns=['x', 'y', 'z', 'lbl'])

    return plot_heatmap(
        'x',
        'y',
        data,
        title=title,
        limx=limx,
        limy=limy,
        limz=limz,
        save=save,
        show=show
    )

    # dfX = pd.DataFrame(data=X)
    # dfX.to_csv("/home/mareike/work/projects/calo-dev/examples/robotaction/dfX.csv", index=False)
    #
    # dfY = pd.DataFrame(data=Y)
    # dfY.to_csv("/home/mareike/work/projects/calo-dev/examples/robotaction/dfY.csv", index=False)
    #
    # dfZ = pd.DataFrame(data=Z)
    # dfZ.to_csv("/home/mareike/work/projects/calo-dev/examples/robotaction/dfZ.csv", index=False)
    #
    # fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
    # fig.update_traces(contours_z=dict(show=True, usecolormap=True,
    #                                   highlightcolor="limegreen", project_z=True))
    #
    # fig.update_layout(title=title, autosize=True,
    #                   width=500, height=500,
    #                   margin=dict(l=65, r=50, b=65, t=90))
    #
    # if save is not None:
    #     fig.write_image(
    #         save,
    #         scale=1
    #     )
    #
    # if show:
    #     fig.show(config=defaultconfig)
    #
    # return fig


def gaussian(
        mean1,
        cov1,
        mean2,
        cov2,
        i=1
):
    gauss1 = Gaussian(mean1, cov1)
    gauss2 = Gaussian(mean2, cov2)

    gaussians = [gauss1, gauss2]

    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x, y)

    xy = np.column_stack([X.flat, Y.flat])
    Z = np.zeros(shape=xy.shape[0])
    for gaussian in gaussians:
        Z += 1 / len(gaussians) * gaussian.pdf(xy)
    Z = Z.reshape(X.shape)

    return x, y, Z, f"test {i}"


def test_hm():
    """plot difference between M(N) and M+1(N+1),
    cmp. https://plotly.com/javascript/reference/heatmap/
    """

    x = np.array([[1, 2, 3, 4, 5]])
    y = np.array([[-1, -2, -3]]).T
    x2 = np.array([[0, 1, 2, 3, 4, 5]])
    y2 = np.array([[0, -1, -2, -3]]).T
    x5 = np.array([[1, 2, 3, 4, 5, 6]])
    y5 = np.array([[-1, -2, -3, -4]]).T
    z = np.array([
        [0, 0.25, 0.25, 0.25, 0],
        [0, 0.25, 0.5, 0.25, 0],
        [0, 0.5, 1, 0.5, 0]
    ])

    plot_heatmap(
            'x',
            'y',
            [
                pd.DataFrame([x, y, z], columns=['x', 'y', 'z']),
                pd.DataFrame([x2, y2, z], columns=['x', 'y', 'z']),
                pd.DataFrame([x5, y5, z], columns=['x', 'y', 'z'])
            ],
            save=os.path.join(locs.examples, 'robotaction', 'tmp_plots', f'testhm.html')
        )


def plot_path(
        xvar,
        yvar,
        p: List,
        title: str = None,
        save: str = None
) -> Figure:

    # generate data points
    d = [
        (
            s[xvar].expectation(),
            s[yvar].expectation(),
            s['xdir_in'].expectation(),
            s['ydir_in'].expectation(),
            f'Step {i}',
            f'Step {i}: {"root" if s.leaf is None or s.tree is None else f"{s.tree}-Leaf#{s.leaf}"}<br>'
            f'PARAM: {param}',
            1
        ) if 'xdir_in' in s and 'ydir_in' in s else (
            first(s[xvar]) if isinstance(s[xvar], set) else s[xvar].lower + abs(s[xvar].upper - s[xvar].lower)/2,
            first(s[yvar]) if isinstance(s[yvar], set) else s[yvar].lower + abs(s[yvar].upper - s[yvar].lower)/2,
            0,
            0,
            f'Step {i}',
            f"Goal",
            1
        ) for i, (s, param) in enumerate(p)
    ]

    # draw scatter points and quivers
    data = pd.DataFrame(
        data=d,
        columns=[xvar, yvar, 'dx', 'dy', 'step', 'lbl', 'size']
    )

    return plot_scatter_quiver(
        xvar,
        yvar,
        data,
        title=title,
        save=save
    )


def plotly_pt(
        pt: Tuple,
        dir: Tuple = None
) -> Any:
    ix, iy = pt

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[ix],
            y=[iy],
            marker=dict(
                symbol='star',
                color='rgb(0,125,0)'
            ),
            fillcolor='rgb(0,125,0)',
            name="Start",
        )
    )

    if dir is not None:
        idx, idy = dir
        fig.add_traces(
            ff.create_quiver(
                [ix],
                [iy],
                [idx],
                [idy],
                scale=.5,
                marker_color='rgb(0,125,0)',
                line_color='rgb(0,125,0)',
                showlegend=False,
                name="Start"
            ).data
        )
    return fig


def plotly_sq(
        area: Tuple,
        lbl: str = "Goal",
        color: str = "0,125,0",
        legend: bool = True
) -> Any:
    gxl, gyl, gxu, gyu = area

    return go.Scatter(
            x=[gxl, gxl, gxu, gxu, gxl],
            y=[gyl, gyu, gyu, gyl, gyl],
            fill="toself",
            marker=dict(
                symbol='star',
                color=f'rgba({color},0.4)'
            ),
            fillcolor=f'rgba({color},0.1)',
            name=lbl,
            showlegend=legend
        )


def plot_pt_sq(
        pt: Tuple,
        area: Tuple
) -> Figure:

    fig = go.Figure()

    fig.add_traces(
        data=plotly_pt(
            pt[:2],
            dir=pt[2:]
        ).data
    )

    # draw square area
    fig.add_trace(
        plotly_sq(area)
    )

    fig.update_coloraxes(
        showscale=False
    )

    fig.update_layout(
        showlegend=False
    )

    return fig


def plot_scatter_quiver(
        xvar,
        yvar,
        data: pd.DataFrame,
        title: str = None,
        save: str = None,
        show: bool = True,
) -> Figure:
    """Plot heatmap or 3D surface plot with plotly
    """
    mainfig = go.Figure()

    # sample colors from continuous color palette to get discrete color sequence
    colors_discr = sample_colorscale(
        px.colors.sequential.dense,
        max(2, len(data)),
        low=.1,  # the first few values are very light and not easy to see on a plot, so we start higher up
        high=1.,
        colortype="rgb"
    )

    colors_discr = colors_discr[:min(len(data), len(colors_discr))]

    # scatter positions
    fig_s = px.scatter(
        data,
        x=xvar,
        y=yvar,
        title=title,
        color_discrete_sequence=colors_discr,
        custom_data=['dx', 'dy', 'lbl'],
        color="step",
        labels=[f'Step {i}' for i in data['step']],
        size='size' if 'size' in data.columns else [1]*len(data),
        width=1700,
        height=1000
    )

    fig_s.update_traces(
        hovertemplate='pos: (%{x:.2f},%{y:.2f})<br>'
                      'dir: (%{customdata[0]:.2f},%{customdata[1]:.2f})<br>'
                      '<extra>%{customdata[2]}</extra>'
    )

    mainfig.add_traces(
        data=fig_s.data
    )

    # draw line connecting points to create a path
    fig_l = px.line(
        data,
        x=xvar,
        y=yvar
    )

    fig_l.update_traces(
        line=dict(
            color='rgba(0,0,0,0.1)'
        )
    )

    mainfig.add_traces(
        data=fig_l.data
    )

    # draw direction arrows (iteration over data as workaround to get differently colored arrows)
    for idx, row in data.iterrows():
        dx, dy = unit([row['dx'], row['dy']])
        f_q = ff.create_quiver(
            [row[xvar]],
            [row[yvar]],
            [dx],
            [dy],
            scale=.5,
            name=row['lbl']
        )

        f_q.update_traces(
            line_color=colors_discr[idx % len(colors_discr)],
            showlegend=False
        )

        mainfig.add_traces(
            data=f_q.data
        )

    # do not draw colorax and legend
    mainfig.layout = fig_s.layout
    mainfig.update_coloraxes(
        showscale=False,
    )

    if save is not None:
        mainfig.write_image(
            save,
            scale=1
        )

    if show:
        mainfig.show(config=defaultconfig)

    return mainfig


def test_plotexamplepath():
    d = []
    for i in range(30):
        d.append(
            [
                i,
                random.randint(i-3, i+3),
                1,
                random.randint(-1, 1),
                f'Step {i}',
                f'Step {i}',
                1
            ]
        )

        data = pd.DataFrame(
        data=d,
        columns=['x', 'y', 'dx', 'dy', 'step', 'lbl', 'size']
    )

    # draw path
    fig = plot_scatter_quiver(
        'x',
        'y',
        data,
        title="plotexamplepath",
        show=False,
        save=os.path.join('/home/mareike/Downloads/test.svg')
    )

    # draw init and goal
    init = (0, 5, 1, 0)
    goal = (28.5, 30, 32, 35)

    fig2 = plot_pt_sq(
        pt=init,
        area=goal

    )

    fig.add_traces(
        data=fig2.data
    )

    return fig


def test_plotexampleheatmap():
    d = [
        [
            [-.25, 0],
            [[.2, -.07], [-.07, .1]],
            [-.5, 1],
            [[.2, .07], [.07, .05]],
            1
        ],[
            [-.25, -.5],
            [[.2, -.07], [-.07, .1]],
            [0., 1],
            [[.2, .07], [.07, .05]],
            2
        ],[
            [-.25, -1],
            [[.2, -.07], [-.07, .1]],
            [.5, 1],
            [[.2, .07], [.07, .05]],
            3
        ],[
            [-.25, -1.5],
            [[.2, -.07], [-.07, .1]],
            [1, 1],
            [[.2, .07], [.07, .05]],
            4
        ],[
            [-.25, -2],
            [[.2, -.07], [-.07, .1]],
            [1.5, 1],
            [[.2, .07], [.07, .05]],
            5
        ]
    ]
    data = pd.DataFrame(
        data=[gaussian(*p) for p in d],
        columns=['x', 'y', 'z', 'lbl']
    )

    plot_heatmap(
        'x',
        'y',
        data,
        title='plotexampleheatmap'
    )


def test_plot_start_goal():
    start = [10, 10, 1, 0]
    goal = [3, 3, 6, 7]

    f = plot_pt_sq(start, goal)
    f.show(config=defaultconfig)


if __name__ == '__main__':
    f = test_plotexamplepath()
    f.show(config=defaultconfig)
    # test_plotexampleheatmap()
    # test_plot_start_goal()
