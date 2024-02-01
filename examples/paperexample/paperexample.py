import math
import os
from typing import Tuple

import numpy as np

import dnutils

import pandas as pd
from calo.logs.logs import init_loggers
from calo.utils import locs
from calo.utils.constants import calologger
from jpt import JPT, infer_from_dataframe
from pandas import DataFrame

logger = dnutils.getlogger(calologger, level=dnutils.DEBUG)


def loaddata(
        d: str,
        dt: str

) -> DataFrame:
    with open(os.path.join(dt, d), 'r') as f:
        cols = list(pd.read_csv(f, delimiter=';', nrows=1))
        df = pd.read_csv(f, usecols=[i for i in cols if i != "id"], delimiter=';', header=0)

        return df


def updatedata(
        d: str,
        dt: str

) -> DataFrame:
    cols = list(pd.read_csv(os.path.join(dt, d), delimiter=';', nrows=1))
    with open(os.path.join(dt, d), 'r') as f:
        usecols = [i for i in cols if i != "id"]
        df = pd.read_csv(f, usecols=usecols, delimiter=';', header=0)

    for ft in cols:
        if not (ft.endswith("_in") and ft.replace("_in", "_out") in cols): continue
        df[ft.replace("_in", "_out")] = df[ft.replace("_in", "_out")] - df[ft]

    return df


def learn(d, dt, fname, tgts):
    vars = infer_from_dataframe(d)

    jpt = JPT(
        variables=vars,
        targets=[v for v in vars if v.name in tgts],
        min_samples_leaf=.1
    )

    jpt.learn(d)
    jpt.postprocess_leaves()

    logger.debug(f'...done! saving to file {os.path.join(dt, fname, f"000-{fname}.tree")}')

    if not os.path.exists(os.path.join(dt, fname)):
        os.mkdir(os.path.join(dt, fname))

    jpt.save(os.path.join(dt, fname, f'000-{fname}.tree'))
    jpt.plot(
        title=fname,
        plotvars=list(jpt.variables),
        filename=f'000-{fname}',
        directory=os.path.join(dt, fname),
        leaffill='#CCDAFF',
        nodefill='#768ABE',
        alphabet=True,
        view=True
    )


if __name__ == "__main__":
    init_loggers(level='debug')

    # use most recently created dataset or create from scratch
    DT = os.path.join(locs.examples, 'paperexample')

    logger.debug(f'running paperexample in {DT}')

    df_d = updatedata("deeprolling.csv", DT)
    learn(df_d, DT, "deeprolling", tgts=[])#["density [kg·m−3]", "deformation []", "dislocationdensity [µm]"])
    df_h = updatedata("heating.csv", DT)
    learn(df_h, DT, "heating", tgts=["deformation_out", "hardness", "dislocationdensity"])
    df_m = updatedata("mechanical.csv", DT)
    learn(df_m, DT, "mechanical", tgts=["color_out", "strength_out"])
    df_t = updatedata("thermal.csv", DT)
    learn(df_t, DT, "thermal", tgts=["color_out", "strength_out"])
