"""This module provides a number of wrapper classes and convenience functions."""
import itertools
import math
import os
from datetime import datetime

import dnutils
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List

import jpt
from calo.core.hypothesis import Step
from calo.utils import locs
from calo.utils.constants import calologger
from calo.utils.constants import xlsHEADER, xlsNUM, xlsDATE
from jpt.base.intervals import ContinuousSet

from jpt.variables import VariableMap

logger = dnutils.getlogger(calologger)


def satisfies(sigma, rho) -> bool:
    """Checks if a state ``sigma`` satisfies the requirement profile ``rho``, i.e. ``φ |= σ``

    :param sigma: a state, e.g. a property-value mapping or position
    :type sigma: jpt.variables.VariableMap
    :param rho: a requirement profile, e.g. a property name-interval, property name-values mapping or position
    :type rho: jpt.variables.VariableMap
    :returns: whether the state satisfies the requirement profile
    :rtype: bool
    """
    # if any property defined in original requirement profile cannot be found in result
    if any(x not in sigma for x in rho.keys()):
        return False

    # value of any resulting variable needs to match interval defined in requirement (if present)
    for k, v in rho.items():
        if k in sigma:
            if isinstance(v, ContinuousSet):
                if not v.contains_value(sigma[k]):
                    return False
                # FIXME: check for expected value or enclosing interval?
                # if not v.contains_interval(ContinuousSet(sigma[k].lower, sigma[k].upper)):
                #     return False
            elif isinstance(v, list):
                # case symbolic variable, v should be list
                if not sigma[k] in v:
                    return False
            else:
                if not sigma[k] == v:
                    return False
    return True


def tovariablemapping(mapping, models) -> jpt.variables.VariableMap:
    if isinstance(mapping, jpt.variables.VariableMap):
        return mapping
    elif all(isinstance(k, jpt.variables.Variable) for k, _ in mapping.items()):
        return VariableMap([(k, v) for k, v in mapping.items()])
    else:
        variables = [v for _, tree in models.items() for v in tree.variables]
        varnames = [v.name for v in variables]
        try:
            # there may be variables with identical names which are different python objects. It is assumed,
            # however, that they are semantically the same, so each of them has to be updated
            return VariableMap([(variables[i], v) for k, v in mapping.items() for i in [i for i, x in enumerate(varnames) if x == k]])
            # return VariableMap([(variables[varnames.index(k)], v) for k, v in mapping.items()])
        except ValueError:
            raise Exception(f'Variable(s) {", ".join([k for k in mapping.keys() if k not in varnames])} are not available in models. Available variables: {varnames}')


# def generate_candidates(query, models) -> List[Step]:
#     """
#
#     :param query: a variable-interval mapping
#     :type query: jpt.variables.VariableMap
#     """
#     return [Step(confs, path, treename, query) for treename, tree in models.items() for idx, (confs, path) in enumerate(tree.reverse({k.name: v for k, v in query.items()}))]


def res(p):
    return os.path.join(locs.resource, p)

def generatemln(data, threshold=10):
    """Expects a list of Example items and generates a template MLN from it as well as training DBs."""
    import pyximport
    pyximport.install()
    from fta.learning.quantiles import Quantiles
    from pracmln import MLN, Database

    featpreds = []
    targetpreds = []

    quantiles = {}
    allfeatures = data[0].features + data[0].targets
    X_input = [x.tosklearn() for x in data]
    X = np.array(X_input)
    X = X.astype(np.float)

    # generate quantiles for discretizing variables that have too many (>threshold) values
    for i, feat in enumerate(allfeatures):
        if len(set(X[:, i])) > threshold:
            quantiles[feat] = Quantiles(np.array(X[:, i], order='C'), epsilon=.0001)

    mln = MLN()
    dbs = []

    for xmpl in data:
        dbs.append(Database(mln))

        # generate fuzzy predicate declarations for features
        for feat in xmpl.x:
            fname = feat.name.replace(' ', '_').replace('-', '').replace('ü', 'ue').replace('ä', 'ae').replace('ö', 'oe')
            pdeclf = '#fuzzy\n{}(sample,{})'.format(fname, fname.lower().replace('_', ''))
            if fname not in featpreds:
                mln << pdeclf
                featpreds.append(fname)
            # add database entry. use original value if variable does not need to be discretized, otherwise use id of corresponding segment in quantile
            # dbs[-1] << '{}({}, "{}")\n'.format(fname, xmpl.identifier, id(quantiles[feat.name].cdf().at(feat.value)) if feat.name in quantiles else feat.value)
            dbs[-1] << '{}({}, "{}")\n'.format(fname, xmpl.identifier, str(quantiles[feat.name].cdf().interval_at(feat.value)).replace('∞', 'inf').replace(',', '-') if feat.name in quantiles else feat.value)
        # generate non-fuzzy predicate declarations for targets
        for tgt in xmpl.t:
            tname = tgt.name.replace(' ', '_').replace('-', '').replace('ü', 'ue').replace('ä', 'ae').replace('ö', 'oe')
            pdeclt = '{}(sample,{})'.format(tname, tname.lower().replace('_', ''))
            if tname not in targetpreds:
                targetpreds.append(tname)
                mln << pdeclt
            # dbs[-1] << '{}({}, "{}")\n'.format(tname, xmpl.identifier, id(quantiles[tgt.name].cdf().at(tgt.value)) if tgt.name in quantiles else tgt.value)
            dbs[-1] << '{}({}, "{}")\n'.format(tname, xmpl.identifier, str(quantiles[tgt.name].cdf().interval_at(tgt.value)).replace('∞', 'inf').replace(',', '-') if tgt.name in quantiles else tgt.value)

    # generate formula (conjunction of all predicates)
    mln << '0. ' + ' ^ '.join(['{}(?s,+?{})'.format(f, f.lower().replace('_', '')) for f in featpreds + targetpreds])
    return mln, dbs


def scatterplot(*args, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*args, **kwargs)
    return ax


def toxls(filename, header, data, sheet='Sheet1') -> bool:
    """
    Generates an .xls file from the given ``data``.

    :param filename: the destination file name
    :param header: the header names
    :param data: the data to write to the .xls file
    :param sheet: the name of the sheet to be created
    :type filename: str
    :type header: list of str
    :type data: 2-dim array of int/str/float
    :type sheet: str
    :returns: True, if successful, False otherwise
    :rtype: bool

    General Usage:

        >>> write(rownum, colnum, label, style= < xlwt.Style.XFStyle object >)

    :Example:

        >>> ws.write(1, 0, datetime.now(), xlsDATE)
        >>> ws.write(2, 2, xlwt.Formula("A3+B3"))
    """
    import xlwt
    wb = xlwt.Workbook(encoding="utf-8")
    ws = wb.add_sheet(sheet)

    for i, h in enumerate(header):
        ws.write(0, i, label=h, style=xlsHEADER)

    for r, row in enumerate(data):
        for c, col in enumerate(row):
            ws.write(r+1, c, label=col, style=xlsNUM)

    wb.save(filename)
    return True


def cov(x) -> np.ndarray:
    # covariance matrix CAUTION! ignoring single nan values
    means = np.nanmean(x, axis=0)
    covmat = np.zeros(shape=(len(means), len(means)))
    for i, j in itertools.product(range(len(means)), repeat=2):
        num = 0
        for r in x:
            if np.isnan(r[i]) or np.isnan(r[j]): continue
            covmat[i][j] += (means[i]-r[i])*(means[j]-r[j])
            num += 1
        covmat[i][j] /= num-1
    return covmat


def pearson(data, use_tgts=True, use_fts=True, removenans=True, ignorenans=True, ignore=None) -> (List[List[str]], np.ndarray):
    # determine pairwise pearson correlation coefficient for features and targets in dataset
    features = data[0].features
    targets = data[0].targets

    if use_fts and use_tgts:
        X_input = [(d.xsklearn() if d.x is not None else []) + (d.tsklearn() if d.t is not None else []) for d in data]
        p = features + targets
    elif use_fts:
        X_input = [d.xsklearn() for d in data if d.x is not None]
        p = features
    else:
        X_input = [d.tsklearn() for d in data if d.t is not None]
        p = targets

    X = np.array(X_input)
    p = np.array(p)

    if ignore is not None:
        colmaskx = np.all(X == ignore, axis=0)
        X = X[:, ~colmaskx]
        p = p[~colmaskx]

    if removenans:
        # remove entire columns that only contain NaN values
        # nancols = np.all(np.isnan(X), axis=0)
        nancols = np.all(np.isnan(X), axis=0)
        X = X[:, ~nancols]
        p = p[~nancols]

    return pcc(X, p, ignorenans=ignorenans)


def pcc(X, fnames, ignorenans=True) -> (List[List[str]], np.ndarray):
    """
    Gets a matrix of data, where each row is a sample, each column a variable and returns two matrices containing the
    feature names and the pearson correlation coefficients of the respective variables
    :param X:           the input data matrix, each row represents a data point, each column a variable
    :param fnames:      the names of the variables, i.e. len(fnames) = len(X[i])
    :param ignorenans:  whether to ignore nan values in the data or not. Caution! When ignoring NaN values, the
                        covariance matrix is built from only the available values
    :return:            two matrices: the feature-feature name and values for the respective PCC
    """

    if ignorenans:
        C = cov(X)
    else:
        C = np.ma.cov(X.transpose())

    if not np.any(C):
        logger.warning('The Covariance matrix for your dataset contains only zeros, you may consider removing '
                       'the variables from your dataset as they do not seem to be significant. Dataset:\n{}'.format(X))

    # create cross product matrix for feature names (to relate pearson matrix to feature combinations)
    pfnames = [' - '.join(x) for x in itertools.product(fnames, repeat=2)]
    m_fnames = [pfnames[i:i + len(fnames)] for i in range(0, len(fnames) ** 2, len(fnames))]

    # generate PCC matrix using formula 5.4 in Information Mining (p.54) (identical to m_pcc for precision 2)
    m_pcc = np.zeros(shape=(len(fnames), len(fnames)))
    for i in range(len(fnames)):
        for j in range(len(fnames)):
            m_pcc[i][j] = pcc_(C, i, j)

    return m_fnames, m_pcc


# calculate PCC (Pearson correlation coefficient)
def pcc_(C, i, j) -> float:
    """
    Calculates the PCC (Pearson correlation coefficient) for the respective variables of the given indices
    :param C:   The covariance matrix for the variables
    :param i:   The index of the first variable to correlate
    :param j:   The index of the second variable to correlate
    :return:    (float) the pcc of the variables with the indices `i` and `j`
    """
    return C[i][j] / math.sqrt(C[i][i] * C[j][j])


def _actions_to_treedata(el, ad) -> dict:
    """For tree visualization; assuming ad is pd.DataFrame, el pd.Series; from NEEM data csv files like:

            id	type	startTime	endTime	duration	success	failure	parent	next	previous	object_acted_on	object_type	bodyPartsUsed	arm	grasp	effort
        Action_IRXOQHDJ	PhysicalTask	1600330068.38499	1600330375.44614	307.061154842377	True
        Action_YGLTFJUW	Transporting	1600330074.30271	1600330375.40287	301.100160360336	True		Action_IRXOQHDJ
        Action_RTGJLPIV	LookingFor	1600330074.64896	1600330074.79814	0.149180889129639	True		Action_YGLTFJUW
        Action_HNLQFJCG	Accessing	1600330075.04209	1600330075.14547	0.103375196456909	True		Action_YGLTFJUW

    """
    tt = f'<b>action:</b> {el["type"]} ({el["id"]})<br>'
    tt += f'<b>duration:</b> {float(el["duration"]):.2f} sec<br>'
    tt += f'<b>timestamps:</b> {float(el["startTime"]):.2f} - {float(el["endTime"]):.2f}<br>'

    if not el["success"] == True:
        tt += f'<b>Error Type:</b> {el["failure"]}'
    else:
        if not pd.isna(el["object_acted_on"]):
            tt += f'<b>object acted on:</b> {el["object_acted_on"]}<br>'
        if not pd.isna(el["bodyPartsUsed"]):
            tt += f'<b>body parts used:</b> {el["bodyPartsUsed"]}<br>'
        if not pd.isna(el["arm"]):
            tt += f'<b>arm:</b> {el["arm"]}<br>'

    return {
        "name": f'{el["type"]}',
        "id": f'{el["id"]}',
        "tooltip": tt,
        "edgetext": f'<b>duration:</b> {float(el["duration"]):.2f} sec<br>',
        "edgetooltip": f'<b>timestamps:</b> {float(el["startTime"]):.2f} - {float(el["endTime"]):.2f}<br>',
        "type": f'{"steelblue" if el["success"] == True else "red"}',
        "showname": True,
        "showedge": False,
        "children": [_actions_to_treedata(c, ad) for _, c in ad[ad['parent'] == el['id']].iterrows()]
    }


def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def angledeg(v1, v2):
    return math.degrees(angle(v1, v2))


def vector(v1, v2):
    return [(a-b) for a, b in zip(v2, v1)]


def unit(v):
    mag = length(v)
    return np.array(v) / mag


def distance(p0, p1):
    return length(vector(p0, p1))


def scale(v, sc):
    return np.array(v) * sc


def add(v1, v2):
    return [(a+b) for a, b in zip(v1, v2)]


def pnt2line(pnt, start, end):
    '''Given a line with coordinates 'init_pos' and 'end' and the
    coordinates of a point 'pnt' the proc returns the shortest
    distance from pnt to the line and the coordinates of the
    nearest point on the line.

    1  Convert the line segment to a vector ('line_vec').
    2  Create a vector connecting init_pos to pnt ('pnt_vec').
    3  Find the length of the line vector ('line_len').
    4  Convert line_vec to a unit vector ('line_unitvec').
    5  Scale pnt_vec by line_len ('pnt_vec_scaled').
    6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
    7  Ensure t is in the range 0 to 1.
    8  Use t to get the nearest location on the line to the end
       of vector pnt_vec_scaled ('nearest').
    9  Calculate the distance from nearest to pnt_vec_scaled.
    10 Translate nearest back to the init_pos/end line.
    Malcolm Kesson 16 Dec 2012

    :param pnt: List or Tuple
    :param start:  List or Tuple
    :param end:  List or Tuple
    :return:
    '''
    if len(pnt) == 2:
        pnt.append(0)
    if len(start) == 2:
        start.append(0)
    if len(end) == 2:
        end.append(0)
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dotproduct(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return dist, nearest
