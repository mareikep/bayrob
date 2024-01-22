import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import dnutils
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from _plotly_utils.colors import sample_colorscale
from jpt.base.intervals import ContinuousSet
from pandas import DataFrame
from plotly.graph_objs import Figure

from calo.utils.constants import calologger
from calo.utils.utils import unit
from jpt import JPT
from jpt.distributions import Gaussian
from jpt.plotting.helpers import color_to_rgb

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


def defaultconfig(fname=None):
    return dict(
        displaylogo=False,
        toImageButtonOptions=dict(
            format='svg',  # one of png, svg, jpeg, webp
            filename=Path(fname).stem if fname is not None else 'calo_plot',
            scale=1  # Multiply title/legend/axis/canvas sizes by this factor
        ),
        # autosizable=True,
        # responsive=True,
        # fillFrame=True,
       #  modeBarButtonsToAdd=[  # allow drawing tools to highlight important regions before downloading snapshot
       #      'drawline',
       #      'drawopenpath',
       #      'drawclosedpath',
       #      'drawcircle',
       #      'drawrect',
       #      'eraseshape'
       # ]
    )


def fig_from_json_file(fname) -> Figure:
    with open(fname) as f:
        return pio.from_json(json.dumps(json.load(f)))


def fig_to_file(fig, fname, ftypes=None) -> None:
    '''Writes figure to file with name `fname`. If multiple extensions are given in ftypes, the same figure is saved
    to multiple file formats.
    '''
    fpath = Path(fname)
    suffix = fpath.suffix
    fname = fpath.stem

    if ftypes is None:
        ftypes = []

    if suffix:
        ftypes.append(suffix)

    ftypes = set([f".{ft}" if not ft.startswith('.') else ft for ft in ftypes])

    for ft in ftypes:
        if ft == '.html':
            logger.debug(f"Saving plot html and json file: {fpath.with_suffix(ft)}...")
            fig.write_json(fpath.with_suffix(ft))
            fig.write_html(
                fpath.with_suffix(ft),
                config=defaultconfig(fname),
                include_plotlyjs="cdn"
            )
        elif ft == '.json':
            logger.debug(f"Saving plot json file: {fpath.with_suffix(ft)}...")
            fig.write_json(fpath.with_suffix(ft))
        else:
            logger.debug(f"Saving plot {ft} file: {fpath.with_suffix(ft)}...")
            fig.write_image(fpath.with_suffix(ft))


def hextorgb(col):
    h = col.strip("#")
    l = len(h)
    if l == 3 or l == 4:  # e.g. "#fff"
        return hextorgb(f'{"".join([(v) * 2 for v in h], )}')
    if l == 6:  # e.g. "#2D6E0F"
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    if l == 8:  # e.g. "#2D6E0F33"
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (round(int(h[6:8], 16) / 255, 2),)


def to_rgb(color, opacity=.6):
    if color.startswith('#'):
        color = hextorgb(color)
        if len(color) == 4:
            opacity = color[-1]
            color = color[:-1]
    elif color.startswith('rgba'):
        color = tuple(map(float, color[5:-1].split(',')))
        opacity = color[-1]
        color = color[:-1]
    elif color.startswith('rgb'):
        color = tuple(map(float, color[4:-1].split(',')))
    return f'rgb{*color,}', f'rgba{*color + (opacity,),}'


def plot_pos(
        path: List,
        title: str = None,
        conf: float = None,
        limx: Tuple = None,
        limy: Tuple = None,
        limz: Tuple = None,
        save: str = None,
        show: bool = True,
        fun: str = "heatmap"
) -> Figure:
    '''
    Plot Heatmap representing distribution of `x_in`, `y_in` variables from a `path`.
    `path` is a list of (state, params) tuples generated in `test_astar_jpt_robotaction.test_astar_cram_path`.
    The
    :param path:
    :param title:
    :param conf:
    :param limx:
    :param limy:
    :param limz:
    :param save:
    :param show:
    :param fun:
    :return:
    '''

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

    return plot_heatmap(
        xvar='x',
        yvar='y',
        data=data,
        title=title,
        limx=limx,
        limy=limy,
        limz=limz,
        save=save,
        show=show,
        fun=fun
    )


def plot_dir(
        path: List,
        title: str = None,
        conf: float = None,
        limx: Tuple = None,
        limy: Tuple = None,
        save: str = None,
        show: bool = True,
        fun: str = "heatmap"
) -> Figure:

    # generate datapoints
    data = pd.DataFrame(
            data=[
                gendata(
                    'xdir_in',
                    'ydir_in',
                    s,
                    p,
                    conf=conf
                ) for i, (s, p) in enumerate(path)
            ],
            columns=['xdir_in', 'ydir_in', 'z', 'lbl']
        )

    return plot_heatmap(
        xvar='xdir_in',
        yvar='ydir_in',
        data=data,
        title=title,
        limx=limx,
        limy=limy,
        save=save,
        show=show,
        fun=fun
    )

def gendata_mixture(
        xvar,
        yvar,
        states,
        limx: Tuple = None,
        limy: Tuple = None,
        priors: List = None,
        numpoints=200
):
    if priors is None:
        priors = [1]*len(states)

    # generate datapoints
    x = np.linspace(limx[0], limx[1], numpoints)
    y = np.linspace(limy[0], limy[1], numpoints)

    X, Y = np.meshgrid(x, y)
    Z = np.array(
        [
            sum([
                s[xvar].pdf(x_) * s[yvar].pdf(y_) * p for s, p in zip(states, priors)
            ]) for x_, y_ in zip(X.ravel(), Y.ravel())
        ]
    ).reshape(X.shape)

    lbl = f'Mixture'

    return x, y, Z, lbl


def gendata(
        xvar,
        yvar,
        state,
        params: Dict = {},
        conf: float = None
):
    '''
    Generates data points
    :param xvar:
    :param yvar:
    :param state:
    :param params:
    :param conf:
    :return:
    '''
    # generate datapoints
    x = state[xvar].pdf.boundaries()
    y = state[yvar].pdf.boundaries()

    X, Y = np.meshgrid(x, y)
    Z = np.array(
        [
            state[xvar].pdf(a) * state[yvar].pdf(b)
            for a, b, in zip(X.ravel(), Y.ravel())
        ]).reshape(X.shape)

    # show only values above a certain threshold, consider lower values as high-uncertainty areas
    if conf is not None:
        Z[Z < conf] = 0.

    # remove or replace by eliminating values > median
    # Z[Z > np.median(Z)] = np.median(Z)

    lbl = f'<b>Leaf#{state.leaf if hasattr(state, "leaf") and state.leaf is not None else "ROOT"}</b> ' \
          f'{"<br>".join([f"<b>{k}:</b> {v}" for k,v in params.items()])}'

    return x, y, Z, lbl


def gendata_multiple(
        vars: List[Tuple],
        states: List,
        params: Dict = {},
        conf: float = None
):
    '''
    Generates data points
    :param vars:
    :param states:
    :param params:
    :param conf:
    :return:
    '''

    data = []
    for (xvar, yvar), state in zip(vars, states):

        # generate datapoints
        x = state[xvar].pdf.boundaries()
        y = state[yvar].pdf.boundaries()

        X, Y = np.meshgrid(x, y)
        Z = np.array(
            [
                state[xvar].pdf(a) * state[yvar].pdf(b)
                for a, b, in zip(X.ravel(), Y.ravel())
            ]).reshape(X.shape)

        # show only values above a certain threshold, consider lower values as high-uncertainty areas
        if conf is not None:
            Z[Z < conf] = 0.

        # remove or replace by eliminating values > median
        # Z[Z > np.median(Z)] = np.median(Z)

        lbl = f'<b>Leaf#{state.leaf if hasattr(state, "leaf") and state.leaf is not None else "ROOT"}</b> ' \
              f'{"<br>".join([f"<b>{k}:</b> {v}" for k,v in params.items()])}'

        data.append([x, y, Z, lbl])
    return data


def plotly_animation(
        data: List[Any],
        names: List[str] = None,
        title: str = None,
        save: str = None,
        show: bool = True,
        showbuttons: bool = True,
        speed: int = 100
) -> Figure:
    '''

    :param names:
    :param data:
    :param title:
    :param save:
    :param show:
    :return:
    '''

    if names is None:
        names = [f'Step {i}' for i in range(len(data))]

    # generate the frames
    frames = [
        go.Frame(
            data=fig_,
            name=name
        ) for fig_, name in zip(data, names)
    ]

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
    )

    if len(frames) > 1:
        # if there are multiple datasets in the Dataframe, create buttons for an animation, otherwise,
        # generate single plot

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
                                    frame_args(speed)
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
                )
            ] + ([
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
                )] if showbuttons else []),
            sliders=sliders
        )
    else:
        if showbuttons:
            fig.update_layout(
                updatemenus=[
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
                ]
            )

    fig.update_layout(
        height=1000,
        width=1000,
        title=title
    )

    if save:
        fig_to_file(fig, save)

    if show:
        fig.show(config=defaultconfig(save))

    return fig


def plot_dists_layered(
        xvar: str,
        yvar: str,
        data: pd.DataFrame,
        limx: Tuple = None,
        limy: Tuple = None,
        save: str = None,
        show: bool = False
) -> Figure:

    mainfig = go.Figure()

    for i, d in data.iterrows():
        mainfig.add_trace(
            go.Heatmap(
                x=d[xvar],
                y=d[yvar].T,
                z=d['z'],
                customdata=d["lbl"] if "lbl" in data.columns and data["lbl"].shape == d["z"].shape else np.full(
                    d['z'].shape, d["lbl"] if "lbl" in data.columns else ""),
                colorscale=px.colors.sequential.dense,
                colorbar=dict(
                    title=f"P({xvar},{yvar})",
                    orientation='v',
                    titleside="top",
                ),
                hovertemplate='x: %{x}<br>'
                              'y: %{y}<br>'
                              'z: %{z}'
                              '<extra>%{customdata}</extra>',
                showscale=False
            )
        )

    mainfig.update_layout(
        width=1000,
        height=1000,
        xaxis=dict(
            range=[*limx]
        ),
        yaxis=dict(
            range=[*limy]
        ),
    )

    if show:
        mainfig.show(config=defaultconfig(save))

    if save:
        fig_to_file(mainfig, save)

    return mainfig


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
        fun: str = "heatmap",
        showbuttons: bool = True
) -> Figure:
    """Plot heatmap (animation) or 3D surface plot with plotly.

    :param xvar: The name of the x-axis of the heatmap and of the respective column in the `data` Dataframe
    :type xvar: str
    :param yvar: The name of the y-axis of the heatmap and of the respective column in the `data` Dataframe
    :type yvar: str
    :param data: Each row (!) consists of an entire dataset, such that multiple rows result in an animation, i.e. for
    an n x m heatmap, each `xvar` cell contains an array of shape (n,), each `yvar` cell contains an array of shape
    (m,) and the `z` cells contain arrays shaped (m,n). May also contain an optional column called `lbl` of
    shape (n,) or (m,n) which serves as custom information when hovering over data points.
    :type data: pd.DataFrame
    :param title: The title of the plot
    :type title: str
    :param limx: The limits of the x-axis. Determined automatically from data if not given
    :type limx: Tuple
    :param limy: The limits of the y-axis. Determined automatically from data if not given
    :type limy: Tuple
    :param save: a full path (including file name) to save the plot to.
    :type save: str
    :param show: whether to automatically open the plot in the default browser.
    :type show: bool
    """

    # determine limits, if not given
    # exclude first element from limit calculations, as it causes the lower probs to be all equal colors
    if limx is None:
        limx = min([df[xvar].min() for _, df in data.iterrows()]), max([df[xvar].max() for _, df in data.iterrows()])

    if limy is None:
        limy = min([df[yvar].min() for _, df in data.iterrows()]), max([df[yvar].max() for _, df in data.iterrows()])

    if limz is None:
        limz_ = min([df['z'].min() for _, df in data.iterrows()]), max([df['z'].max() for _, df in data.iterrows()])
    else:
        limz_ = limz

    addargs = {} if limz is None else {"zmin" if fun == "heatmap" else "cmin": limz_[0], "zmax" if fun == "heatmap" else "cmax": limz_[1]}
    fun = {"heatmap": go.Heatmap, "surface": go.Surface}.get(fun, go.Heatmap)

    frames = [
        (
            fun(
                x=d[xvar],
                y=d[yvar].T,
                z=d['z'] if limz is None else np.clip(d['z'], *limz_),
                customdata=d["lbl"] if "lbl" in data.columns and data["lbl"].shape == d["z"].shape else np.full(d['z'].shape, d["lbl"] if "lbl" in data.columns else ""),
                colorscale=px.colors.sequential.dense,
                colorbar=dict(
                    title=f"P({xvar},{yvar})",
                    orientation='v',
                    titleside="top",
                ),
                hovertemplate='x: %{x}<br>'
                              'y: %{y}<br>'
                              'z: %{z}'
                              '<extra>%{customdata}</extra>',
                **addargs
            )
        ) for i, d in data.iterrows()
    ]

    fig = plotly_animation(
        data=frames,
        show=False,
        showbuttons=showbuttons,
        title=None if title is False else title,
        speed=100
    )

    fig.update_layout(
        xaxis=dict(
            title=xvar,
            side='top',
            range=[*limx]
        ),
        yaxis=dict(
            title=yvar,
            range=[*limy]
        ),
        # paper_bgcolor="black",
        # plot_bgcolor="black"
    )

    if save:
        fig_to_file(fig, save)

    if show:
        fig.show(config=defaultconfig(save))

    return fig


def plot_tree_dist(
    tree: JPT,
    qvars: dict = None,
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
    Z = []
    for y_ in y:
        z_ = []
        for x_ in x:
            d = {qvarx: x_, qvary: y_}
            if qvars is not None:
                d.update(qvars)
            z_.append(tree.pdf(tree.bind(d)))
        Z.append(z_)

    Z = np.array(Z)

    lbl = np.full(Z.shape, f'Some random label')

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

def pathdata(
        xvar,
        yvar,
        p: List,
        exp: bool = False
) -> List:
    return [
        (
            s[xvar].expectation() if exp else np.mean([s[xvar].mpe()[0].lower, s[xvar].mpe()[0].upper]),  # x
            s[yvar].expectation() if exp else np.mean([s[yvar].mpe()[0].lower, s[yvar].mpe()[0].upper]),  # y
            s['xdir_in'].expectation() if exp else np.mean([s['xdir_in'].mpe()[0].lower, s['xdir_in'].mpe()[0].upper]),  # dx
            s['ydir_in'].expectation() if exp else np.mean([s['ydir_in'].mpe()[0].lower, s['ydir_in'].mpe()[0].upper]),  # dy
            f'Step {i}: {param.get("action")}({",".join([f"{k}: {v}" for k,v in param.items() if k != "action"])})'.ljust(50),
            f'<b>Step {i}: {"root" if s.leaf is None or s.tree is None else f"{s.tree}-Leaf#{s.leaf}"}</b><br>'
            f'<b>MPEs:</b><br>{"<br>".join(f"{k}: {v.mpe()[0]}" for k, v in s.items())}<br>'
            f'<b>Expectations:</b><br>{"<br>".join(f"{k}: {v.expectation()}" for k, v in s.items())}',  # lbl
            1
        ) for i, (s, param) in enumerate(p) if {'x_in', 'y_in'}.issubset(set(s.keys()))
    ]

def plot_path(
        xvar,
        yvar,
        p: List,
        obstacles: List = None,
        title: str = None,
        save: str = None,
        show: bool = False,
) -> Figure:

    # generate data points
    d = pathdata(xvar, yvar, p, exp=False)

    # draw scatter points and quivers
    data = pd.DataFrame(
        data=d,
        columns=[xvar, yvar, 'dx', 'dy', 'step', 'lbl', 'size']
    )

    fig = go.Figure()

    # plot obstacles in background
    if obstacles is not None:
        for (o, on) in obstacles:
            fig.add_trace(
                plotly_sq(o, lbl=on, color='rgb(15,21,110)', legend=False))


    fig_ = plot_scatter_quiver(
        xvar,
        yvar,
        data,
        title=title,
        save=save,
    )

    fig.add_traces(fig_.data)
    fig.layout = fig_.layout
    fig.update_layout(
        height=1000,
        width=1200,
        title=title
    )

    if show:
        fig.show(config=defaultconfig(save))

    return fig


def plotly_pt(
        pt: Tuple,
        dir: Tuple = None,
        name: str = None,
        color: str = "rgb(15,21,110)"
) -> Any:
    ix, iy = pt
    rgb, rgba = to_rgb(color, opacity=1)
    _, rgba1 = to_rgb(color, opacity=0.1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[ix],
            y=[iy],
            marker=dict(
                symbol='star',
                color=rgb
            ),
            fillcolor=rgba1,
            name=name,
            showlegend=False
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
        color: str = "rgb(15,21,110)",
        legend: bool = True
) -> Any:
    gxl, gyl, gxu, gyu = area
    rgb, rgba = to_rgb(color, opacity=1)
    _, rgba1 = to_rgb(color, opacity=0.1)

    return go.Scatter(
            x=[gxl, gxl, gxu, gxu, gxl],
            y=[gyl, gyu, gyu, gyl, gyl],
            fill="toself",
            textposition='top right',
            textfont=dict(color=rgb),
            marker=dict(
                symbol='circle',
                color=rgb,
                line=dict(
                    color=rgb,
                    width=5
                )
            ),
            fillcolor=rgba1,
            name=lbl,
            showlegend=legend,
            mode="lines+text",
            text=[lbl, None, None, None, None]
        )


def plot_pt_sq(
        pt: Tuple,
        area: Tuple
) -> Figure:

    fig = go.Figure()

    fig.add_traces(
        data=plotly_pt(
            pt[:2],
            dir=pt[2:],
            name="Start"
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
        show: bool = False,
) -> Figure:
    """Plot heatmap or 3D surface plot with plotly
    """
    mainfig = go.Figure()

    # sample colors from continuous color palette to get discrete color sequence
    colors_discr = sample_colorscale(
        px.colors.sequential.dense,
        max(2, len(data)+2),
        low=.2,  # the first few values are very light and not easy to see on a plot, so we start higher up
        high=1.,
        colortype="rgb"
    )

    colors_discr = colors_discr[:min(len(data)+1, len(colors_discr))]
    colors_discr_a = [color_to_rgb(c, 1)[1] for c in colors_discr]

    # scatter positions
    fig_s = go.Scatter(
        x=data[xvar],
        y=data[yvar],
        marker=dict(
            color=list(range(len(colors_discr_a))),  # set color equal to a variable
            colorscale=colors_discr_a,
            size=15,
            line=dict(
                color='rgba(255,255,255,1.0)',
                # color=list(range(len(colors_discr))),
                # colorscale=colors_discr,
                width=2
            )
        ),
        customdata=data[['dx', 'dy', 'lbl']],
        hovertemplate='pos: (%{x:.2f},%{y:.2f})<br>'
                      'dir: (%{customdata[0]:.2f},%{customdata[1]:.2f})<br>'
                      '<extra>%{customdata[2]}</extra>',
        mode="markers",
        text=data['step'],
        showlegend=False,
    )

    mainfig.add_trace(fig_s)

    # align x- and y-axis (scale ticks)
    mainfig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1),
        title=title
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
    # dxx = []
    # dyy = []
    # for idx, row in data.iterrows():
    #     dx, dy = unit([row['dx'], row['dy']])
    #     dxx.append(dx)
    #     dyy.append(dy)
    #
    # f_q = ff.create_quiver(
    #     data[xvar],
    #     data[yvar],
    #     dxx,
    #     dyy,
    #     scale=3,
    #     customdata=data.loc[list(data.index.repeat(3)) + list(data.index.repeat(4)), ['dx', 'dy', 'lbl']], #data[['dx', 'dy', 'lbl']],
    #     hovertemplate='pos: (%{x:.2f},%{y:.2f})<br>'
    #                   'dir: (%{customdata[0]:.2f},%{customdata[1]:.2f})<br>'
    #                   '<extra>%{customdata[2]}</extra>',
    #     showlegend=False
    # )
    #
    # f_q.update_traces(
    #     line_color=colors_discr[idx % len(colors_discr)],
    #     showlegend=False
    # )
    #
    # mainfig.add_traces(
    #     data=f_q.data
    # )

    for idx, row in data.iterrows():
        dx, dy = unit([row['dx'], row['dy']])
        f_q = ff.create_quiver(
            [row[xvar]],
            [row[yvar]],
            [dx],
            [dy],
            scale=3,
            name=f"Step {idx}",
            customdata=data.loc[[idx] * 7, ['dx', 'dy', 'lbl']],
            hovertemplate='pos: (%{x:.2f},%{y:.2f})<br>'
                          'dir: (%{customdata[0]:.2f},%{customdata[1]:.2f})<br>'
                          '<extra>%{customdata[2]}</extra>',
        )

        f_q.update_traces(
            line_color=colors_discr[idx % len(colors_discr)],
            showlegend=False
        )

        mainfig.add_traces(
            data=f_q.data
        )

    # do not draw colorax and legend
    # mainfig.layout = fig_s.layout

    if save is not None:
        fig_to_file(mainfig, save)

    if show:
        mainfig.show(config=defaultconfig(save))

    return mainfig


def filter_dataframe(
        df: pd.DataFrame,
        constraints
) -> pd.DataFrame:

    # constraints is a list of 3-tuples: ('<column name>', 'operator', value)
    constraints_ = []
    for var, val in constraints.items():
        if isinstance(val, ContinuousSet):
            for v, op in [(val.lower, ">="), (val.upper, "<=")]:
                constraints_.append(f'(`{var}` {op} {v})')
        elif isinstance(val, (list, set)):
            constraints_.append("(" + '|'.join([f'`{var}` == "{v}"' for v in val]) + ")")
        elif isinstance(val, str):
            constraints_.append(f'(`{var}` == "{val}")')
        else:
            constraints_.append(f'(`{var}` == {val})')

    s = ' & '.join(constraints_)
    logger.debug('Extracting dataset using query: ', s)

    if s == "":
        df_ = df
    else:
        df_ = df.query(s)

    logger.debug('Returned subset of shape:', df_.shape)
    return df_


def plot_data_subset(
        df,
        xvar,
        yvar,
        constraints,
        limx=None,
        limy=None,
        save=None,
        show=False,
        plot_type='scatter',
        normalize=False,
        color='rgb(15,21,110)'
):
    if limx is None:
        limx = [df[xvar].min(), df[xvar].max()]

    if limy is None and yvar is not None:
        limy = [df[yvar].min(), df[yvar].max()]

    df_ = filter_dataframe(df, constraints)

    # extract rgb colors from given hex, rgb or rgba string
    rgb, rgba = color_to_rgb(color)

    if df_.shape[0] == 0:
        logger.warning('EMPTY DATAFRAME!')
        return

    print(df_.shape[0])

    fig_s = go.Figure()
    if plot_type == "scatter":
        fig_s.add_trace(
            go.Scatter(
                x=df_[xvar],
                y=df_[yvar],
                marker=dict(
                    color=rgba,
                    size=3,
                    line=dict(
                        color=rgb,
                        width=1
                    )
                ),
                fillcolor=rgba,
                mode="markers",
                showlegend=False
            )
        )

        fig_s.update_layout(
            xaxis=dict(
                range=limx
            ),
            yaxis=dict(
                range=limy
            ),
        )
    elif plot_type == "histogram":
        x = df_[xvar].value_counts().index.tolist()
        y = list(df_[xvar].value_counts(normalize=normalize))
        fig_s.add_trace(
            go.Bar(
                x=x,
                y=y,
                text=y,
                orientation='v',
                marker=dict(
                    color=rgba,
                    line=dict(color=rgb, width=3)
                )
            )
        )
    else:
        logger.error("Can only plot scatter or histogram")
        return

    fig_s.update_layout(
        xaxis_title=xvar,
        yaxis_title=yvar if plot_type == "scatter" else f"count({xvar})",
        showlegend=False,
        width=1000,
        height=1000,
        yaxis=dict(
            range=(0, 1)
        ) if normalize else {}
    )

    if show:
        fig_s.show(config=defaultconfig(save))

    if save:
        fig_to_file(fig_s, save)

    return fig_s


# ==================================== TESTS ===========================================================================

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

    return x, y, Z, np.full(x.shape, f"test {i}")


def plot_multiple_dists(
        gaussians,
        limx: Tuple = None,
        limy: Tuple = None,
        save: str = None,
        show: bool = False
):
    mainfig = go.Figure()

    for g in gaussians:
        # generate datapoints
        x = g[0].pdf.boundaries()
        y = g[1].pdf.boundaries()

        X, Y = np.meshgrid(x, y)
        Z = np.array(
            [
                g[0].pdf(a) * g[1].pdf(b)
                for a, b, in zip(X.ravel(), Y.ravel())
            ]).reshape(X.shape)

        mainfig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=Z,
                colorscale='dense',
                contours_z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="limegreen",
                    project_z=True
                ),
                showscale=False
            ),
        )

    mainfig.update_layout(
        width=1000,
        height=1000,
        xaxis=dict(
            range=[*limx]
        ),
        yaxis=dict(
            range=[*limy]
        ),
    )

    if show:
        mainfig.show(config=defaultconfig(save))

    if save:
        fig_to_file(mainfig, save)


if __name__ == "__main__":
    robot_positions = {
        "kitchen_unit": [
            (30, 72, [15, .07], [.07, 1], 0, 1, 30, 15),
        ],
        "fridge": [
            (63, 72, [5, -.07], [-.07, 1], .7, .7, 20, 5)
        ],
        "stove": [
            (72, 50, [.2, -.07], [-.07, 10], 1, 0, 20, 15),
        ],
        "kitchen_island": [
            (30, 55, [15, .07], [.07, 1], 0, -1, 30, 15),
            (55, 40, [0.2, -.07], [-.07, 10], -1, 0, 20, 7)
        ],
        "chair2": [
            (55, 15, [0.2, -.07], [-.07, 10], -1, 0, 20, 5)
        ]
    }

    gaussians = [(name, Gaussian([mx_, my_], [stdx_, stdy_])) for name, query in robot_positions.items() for mx_, my_, stdx_, stdy_, dx, dy, angle, nums in query]
    dfs = [(name, DataFrame(data=g.sample(50), columns=['x', 'y'])) for name, g in gaussians]
    colors = ["#FF0000", "#FFC900", "#00FF00", "#00D4FF", "#0014FF", "#8E00FF", "#111111"]
    colors = {n[0]: c for n, c in zip(dfs, colors)}

    fig = go.Figure()
    for i, (c, df) in enumerate(dfs):
        fig.add_trace(
            go.Scatter(
                x=df['x'],
                y=df['y'],
                marker=dict(
                    symbol='circle',
                    color=colors[c]
                ),
                mode="markers",
                fillcolor=colors[c],
                name=c,
            )
        )

    fig.show()