import os
import random
from typing import List, Tuple, Dict

import plotly.graph_objects as go
import plotly.express as px
from _plotly_utils.colors import sample_colorscale
from plotly.graph_objs import Figure
import plotly.figure_factory as ff
from plotly.graph_objs.heatmap import Hoverlabel
from plotly.graph_objs.layout.shape import Label

import numpy as np
import pandas as pd
from calo.utils import locs
from calo.utils.utils import unit
from dnutils import first
from jpt.distributions import Gaussian


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
    data = [
        pd.DataFrame(
            data=[
                gendata(
                    'x_in',
                    'y_in',
                    s,
                    p,
                    conf=conf
                )
            ],
            columns=['x', 'y', 'z', 'lbl']
        ) for i, (s, p) in enumerate(path)
    ]

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

    return X, Y, Z, lbl


# def gendata_out(
#         xvar,
#         yvar,
#         state,
#         params,
#         conf=None,
#         failifnotpresent: bool=True
# ):
#     # check if xvar and yvar have corresponding _out variables in state.leaf
#     xvar_out = xvar.replace('_in', '_out')
#     yvar_out = yvar.replace('_in', '_out')
#
#     if state.leaf is None or state.tree is None:
#         # the very first state of a sequence does not have a leaf attached, therefore we can only use the distributions
#         # stored in the state directly (i.e. the '_in'-distributions of the initstate)
#         return gendata(
#             xvar,
#             yvar,
#             state,
#             params,
#             conf=conf
#         )
#
#     # if corresponding _out variables is not present, either throw an error or only use the _in variable instead
#     # it might make sense to set failifnotpresent to false, when gendata_out is called in a loop and some states
#     # may be the result of turn actions, whose leaves do not contain out_variables for dir
#     if xvar_out not in state.tree.leaves[state.leaf].distributions:
#         if failifnotpresent:
#             raise ValueError(f'Variable {xvar_out} is not present in state distributions. Available variables are: {", ".join([v.name for v in state.tree.leaves[state.leaf].distributions])}')
#         # else:
#         #     return gendata(
#         #         xvar,
#         #         yvar,
#         #         state,
#         #         params,
#         #         conf=conf
#         #     )
#
#     if yvar_out not in state.tree.leaves[state.leaf].distributions:
#         if failifnotpresent:
#             raise ValueError(f'Variable {yvar_out} is not present in state distributions. Available variables are: {", ".join([v.name for v in state.tree.leaves[state.leaf].distributions])}')
#         # else:
#         #     return gendata(
#         #         xvar,
#         #         yvar,
#         #         state,
#         #         params,
#         #         conf=conf
#         #     )
#
#     # generate new distributions by performing distribution addition (and reducing complexity to complexity of
#     # previous, unaltered distribution to keep resulting distribution manageable)
#     if xvar_out not in state.tree.leaves[state.leaf].distributions:
#         xdist = state[xvar]
#     else:
#         xdist = state[xvar] + state.tree.leaves[state.leaf].distributions[xvar_out]
#         if hasattr(xdist, 'approximate'):
#             xdist = xdist.approximate(n_segments=len(state[xvar].pdf.functions))
#
#     if yvar_out not in state.tree.leaves[state.leaf].distributions:
#         ydist = state[yvar]
#     else:
#         ydist = state[yvar] + state.tree.leaves[state.leaf].distributions[yvar_out]
#         if hasattr(ydist, 'approximate'):
#             ydist = ydist.approximate(n_segments=len(state[yvar].pdf.functions))
#
#     # generate datapoints
#     x = xdist.pdf.boundaries()
#     y = ydist.pdf.boundaries()
#
#     X, Y = np.meshgrid(x, y)
#     Z = np.array(
#         [
#             xdist.pdf(x) * ydist.pdf(y)
#             for x, y, in zip(X.ravel(), Y.ravel())
#         ]).reshape(X.shape)
#
#     # show only values above a certain threshold, consider lower values as high-uncertainty areas
#     if conf is not None:
#         Z[Z < conf] = 0.
#
#     # remove or replace by eliminating values > median
#     Z[Z > np.median(Z)] = np.median(Z)
#
#     return X, Y, Z, params


def plot_heatmap(
        xvar: str,
        yvar: str,
        data: List[pd.DataFrame],
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
        limx = min([df[xvar][0].min() for df in data]), max([df[xvar][0].max() for df in data])

    if limy is None:
        limy = min([df[yvar][0].min() for df in data]), max([df[yvar][0].max() for df in data])

    if limz is None:
        limz = min([df["z"][0].min() for df in data]), max([df["z"][0].max() for df in data])

    # generate the frames
    frames = [
        go.Frame(
            data=go.Heatmap(
                x=d[xvar][0][0],  # x[0],
                y=d[yvar][0].T[0],  # y.T[0],
                z=d['z'][0],  # z,
                zmin=limz[0],
                zmax=limz[1],
                customdata=np.full(d['z'][0].shape, d["lbl"]),
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
        ) for i, d in enumerate(data)
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
                include_plotlyjs="cdn"
            )
        else:
            fig.write_image(save)

    if show:
        fig.show()

    return fig


# def plot_xyvars(
#         xvar: str,
#         yvar: str,
#         path: List,
#         title: str = None,
#         conf: float = None,
#         limx: Tuple = None,
#         limy: Tuple = None,
#         limz: Tuple = None,
#         save: str = None,
#         show: bool = True,
#         animation: bool = False
# ) -> None:
#     """ONLY FOR GRIDWORLD DATA
#     """
#     from matplotlib import pyplot as plt
#     import matplotlib.animation as animation
#     cmap = 'BuPu'  # viridis, Blues, PuBu, 0rRd, BuPu
#
#     def gendata(frame):
#         s, p = path[frame]
#
#         # generate datapoints
#         x = s[xvar].pdf.boundaries()
#         y = s[yvar].pdf.boundaries()
#
#         X, Y = np.meshgrid(x, y)
#         Z = np.array(
#             [
#                 s[xvar].pdf(x) * s[yvar].pdf(y)
#                 for x, y, in zip(X.ravel(), Y.ravel())
#             ]).reshape(X.shape)
#
#         # show only values above a certain threshold, consider lower values as high-uncertainty areas
#         if conf is not None:
#             Z[Z < conf] = 0.
#
#         # remove or replace by eliminating values > median
#         Z[Z > np.median(Z)] = np.median(Z)
#
#         return X, Y, Z
#
#     def update(frame):
#         X, Y, Z = gendata(frame)
#
#         # determine limits
#         zmin = ifnone(limz, Z.min(), lambda l: l[0])
#         zmax = ifnone(limz, Z.max(), lambda l: l[1])
#
#         # generate heatmap
#         ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=zmin, vmax=zmax)
#
#     if animation:
#
#         # init plot
#         fig, ax = plt.subplots(num=1, clear=True)
#         fig.patch.set_facecolor('#D6E7F8')  # set bg color around the plot area (royal purple)
#         ax.set_facecolor('white')  # set bg color of plot area (dark purple)
#
#         X, Y, Z = gendata(0)
#
#         c = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=limz[0], vmax=limz[1])
#         fig.colorbar(c, ax=ax)
#         fig.suptitle(title)
#         fig.canvas.manager.set_window_title(f'Belief State: P({xvar}/{yvar})')
#
#         ax.set_title(f'P({xvar},{yvar})')
#
#         # setting the limits of the plot to the limits of the data
#         ax.axis([limx[0], limx[1], limy[0], limy[1]])
#         ax.set_xlabel(rf'${xvar}$')
#         ax.set_ylabel(rf'${yvar}$')
#
#         ani = animation.FuncAnimation(
#             fig=fig,
#             func=update,
#             frames=len(path),  # length of the animation
#             interval=30  # time in milliseconds between drawing of two frames
#         )
#
#         if save:
#             # plt.savefig(f'{i:03}-{save}')
#             ani.save(filename=f'{save}.mp4', writer="ffmpeg")
#
#         if show:
#             plt.show()
#     else:
#
#         for i, _ in enumerate(path):
#             X, Y, Z, p = gendata(i)
#
#             # init plot
#             fig, ax = plt.subplots(num=1, clear=True)
#             fig.patch.set_facecolor('#D6E7F8')  # set bg color around the plot area (royal purple)
#             ax.set_facecolor('white')  # set bg color of plot area (dark purple)
#
#             # determine limits
#             xmin = ifnone(limx, X.min(), lambda l: l[0])
#             xmax = ifnone(limx, X.max(), lambda l: l[1])
#             ymin = ifnone(limy, Y.min(), lambda l: l[0])
#             ymax = ifnone(limy, Y.max(), lambda l: l[1])
#             zmin = ifnone(limz, Z.min(), lambda l: l[0])
#             zmax = ifnone(limz, Z.max(), lambda l: l[1])
#
#             # generate heatmap
#             c = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=zmin, vmax=zmax)
#
#             ax.set_title(f'P({xvar},{yvar})')
#
#             # setting the limits of the plot to the limits of the data
#             ax.axis([xmin, xmax, ymin, ymax])
#             ax.set_xlabel(rf'${xvar}$')
#             ax.set_ylabel(rf'${yvar}$')
#             fig.colorbar(c, ax=ax)
#             fig.suptitle(title)
#             fig.canvas.manager.set_window_title(f'Belief State at Step{i}: P({xvar}/{yvar}), params: {p}')
#
#             if save:
#                 plt.savefig(f'{i:03}-{save}')
#
#             if show:
#                 plt.show()


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

    return X, Y, Z, f"test {i}"


def testhm():
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

    plot_heatmap_test(
            'x',
            'y',
            [
                [x, y, z],
                [x2, y2, z],
                [x5, y5, z],
            ],
            save=os.path.join(locs.examples, 'robotaction', 'tmp_plots', f'testhm.html')
        )


def plot_path(
        xvar,
        yvar,
        p: List,
        title: None,
        save: str = None
) -> None:

    # generate data points
    d = [
        (
            s[xvar].expectation(),
            s[yvar].expectation(),
            s['xdir_in'].expectation(),
            s['ydir_in'].expectation(),
            f'{i}-Leaf#{s.leaf if s.leaf is not None else "ROOT"} '
            f'({s[xvar].expectation():.2f},{s[yvar].expectation():.2f}): '
            f'({s["xdir_in"].expectation():.2f},{s["ydir_in"].expectation():.2f})'
            f'PARAM: {param}',
            1
        ) if 'xdir_in' in s and 'ydir_in' in s else (
            first(s[xvar]) if isinstance(s[xvar], set) else s[xvar].lower + abs(s[xvar].upper - s[xvar].lower)/2,
            first(s[yvar]) if isinstance(s[yvar], set) else s[yvar].lower + abs(s[yvar].upper - s[yvar].lower)/2,
            0,
            0,
            f"Goal",
            1
        ) for i, (s, param) in enumerate(p)
    ]

    # draw scatter points and quivers
    data = pd.DataFrame(data=d, columns=[xvar, yvar, 'dx', 'dy', 'Step', 'size'])

    plot_scatter_quiver(
        xvar,
        yvar,
        data,
        lbl='Step',
        title=title,
        save=save
    )


def plot_pt_sq(
        pt: Tuple,
        area: Tuple
) -> Figure:
    ix, iy, idx, idy = pt
    gxl, gyl, gxu, gyu = area

    fig = px.scatter(
        [[ix, iy]],
        x=0,
        y=1,
        color_discrete_sequence=['rgb(0,125,0)'],
        labels=["Start"]
    )

    f_q = ff.create_quiver(
        [ix],
        [iy],
        [idx],
        [idy],
        scale=.5,
    )

    f_q.update_traces(
        line_color='rgb(0,125,0)',
        showlegend=False
    )

    fig.add_traces(
        data=f_q.data
    )

    # Add a shape whose x and y coordinates refer to the domains of the x and y axes
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=gxl, x1=gxu, y0=gyl, y1=gyu,
        label=Label(
            text='Goal',
            textposition="middle center"
        ),
        fillcolor='rgb(0,125,0)',
        opacity=0.1
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
        data: List,
        lbl: str = 'label',
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
        low=0.0,
        high=.9,
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
        hover_data=lbl,
        color=lbl,
        labels=lbl,
        size='size' if 'size' in data.columns else [1]*len(data),
        width=2000,
        height=1000
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
            name=row[lbl]
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
        mainfig.show()

    return mainfig


def plotexamplepath():
    d = []
    for i in range(30):
        d.append([i, random.randint(i-3, i+3), 1, random.randint(-1, 1), f'Step {i}', 1])

        data = pd.DataFrame(
        data=d,
        columns=['x', 'y', 'dx', 'dy', 'Step', 'size']
    )

    plot_scatter_quiver(
        'x',
        'y',
        data,
        lbl='Step',
        title="plotexamplepath",
        save=os.path.join('/home/mareike/Downloads/test.svg')
    )


def plotexampleheatmap():
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
    data = [pd.DataFrame(data=[gaussian(*p)], columns=['x', 'y', 'z', 'lbl']) for p in d]

    plot_heatmap(
        'x',
        'y',
        data,
        title='plotexampleheatmap'
    )


def plot_start_goal():
    start = [10, 10, 1, 0]
    goal = [3, 3, 6, 7]

    plot_pt_sq(start, goal)


if __name__ == '__main__':
    # plotexamplepath()
    plotexampleheatmap()
    # plot_start_goal()
