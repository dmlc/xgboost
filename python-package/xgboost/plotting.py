# coding: utf-8
# pylint: disable=too-many-locals, too-many-arguments, invalid-name,
# pylint: disable=too-many-branches
"""Plotting Library."""
from __future__ import absolute_import

import re
from io import BytesIO
import numpy as np
from .core import Booster
from .sklearn import XGBModel


def plot_importance(booster, ax=None, height=0.2,
                    xlim=None, ylim=None, title='Feature importance',
                    xlabel='F score', ylabel='Features',
                    importance_type='weight', max_num_features=None,
                    grid=True, show_values=True, **kwargs):

    """Plot importance based on fitted trees.

    Parameters
    ----------
    booster : Booster, XGBModel or dict
        Booster or XGBModel instance, or dict taken by Booster.get_fscore()
    ax : matplotlib Axes, default None
        Target axes instance. If None, new figure and axes will be created.
    grid : bool, Turn the axes grids on or off.  Default is True (On).
    importance_type : str, default "weight"
        How the importance is calculated: either "weight", "gain", or "cover"
        "weight" is the number of times a feature appears in a tree
        "gain" is the average gain of splits which use the feature
        "cover" is the average coverage of splits which use the feature
            where coverage is defined as the number of samples affected by the split
    max_num_features : int, default None
        Maximum number of top features displayed on plot. If None, all features will be displayed.
    height : float, default 0.2
        Bar height, passed to ax.barh()
    xlim : tuple, default None
        Tuple passed to axes.xlim()
    ylim : tuple, default None
        Tuple passed to axes.ylim()
    title : str, default "Feature importance"
        Axes title. To disable, pass None.
    xlabel : str, default "F score"
        X axis title label. To disable, pass None.
    ylabel : str, default "Features"
        Y axis title label. To disable, pass None.
    show_values : bool, default True
        Show values on plot. To disable, pass False.
    kwargs :
        Other keywords passed to ax.barh()

    Returns
    -------
    ax : matplotlib Axes
    """
    # TODO: move this to compat.py
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('You must install matplotlib to plot importance')

    if isinstance(booster, XGBModel):
        importance = booster.get_booster().get_score(importance_type=importance_type)
    elif isinstance(booster, Booster):
        importance = booster.get_score(importance_type=importance_type)
    elif isinstance(booster, dict):
        importance = booster
    else:
        raise ValueError('tree must be Booster, XGBModel or dict instance')

    if len(importance) == 0:
        raise ValueError('Booster.get_score() results in empty')

    tuples = [(k, importance[k]) for k in importance]
    if max_num_features is not None:
        tuples = sorted(tuples, key=lambda x: x[1])[-max_num_features:]
    else:
        tuples = sorted(tuples, key=lambda x: x[1])
    labels, values = zip(*tuples)

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ylocs = np.arange(len(values))
    ax.barh(ylocs, values, align='center', height=height, **kwargs)

    if show_values is True:
        for x, y in zip(values, ylocs):
            ax.text(x + 1, y, x, va='center')

    ax.set_yticks(ylocs)
    ax.set_yticklabels(labels)

    if xlim is not None:
        if not isinstance(xlim, tuple) or len(xlim) != 2:
            raise ValueError('xlim must be a tuple of 2 elements')
    else:
        xlim = (0, max(values) * 1.1)
    ax.set_xlim(xlim)

    if ylim is not None:
        if not isinstance(ylim, tuple) or len(ylim) != 2:
            raise ValueError('ylim must be a tuple of 2 elements')
    else:
        ylim = (-1, len(values))
    ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(grid)
    return ax


_NODEPAT = re.compile(r'(\d+):\[(.+)\]')
_LEAFPAT = re.compile(r'(\d+):(leaf=.+)')
_EDGEPAT = re.compile(r'yes=(\d+),no=(\d+),missing=(\d+)')
_EDGEPAT2 = re.compile(r'yes=(\d+),no=(\d+)')


def _parse_node(graph, text):
    """parse dumped node"""
    match = _NODEPAT.match(text)
    if match is not None:
        node = match.group(1)
        graph.node(node, label=match.group(2), shape='circle')
        return node
    match = _LEAFPAT.match(text)
    if match is not None:
        node = match.group(1)
        graph.node(node, label=match.group(2), shape='box')
        return node
    raise ValueError('Unable to parse node: {0}'.format(text))


def _parse_edge(graph, node, text, yes_color='#0000FF', no_color='#FF0000'):
    """parse dumped edge"""
    try:
        match = _EDGEPAT.match(text)
        if match is not None:
            yes, no, missing = match.groups()
            if yes == missing:
                graph.edge(node, yes, label='yes, missing', color=yes_color)
                graph.edge(node, no, label='no', color=no_color)
            else:
                graph.edge(node, yes, label='yes', color=yes_color)
                graph.edge(node, no, label='no, missing', color=no_color)
            return
    except ValueError:
        pass
    match = _EDGEPAT2.match(text)
    if match is not None:
        yes, no = match.groups()
        graph.edge(node, yes, label='yes', color=yes_color)
        graph.edge(node, no, label='no', color=no_color)
        return
    raise ValueError('Unable to parse edge: {0}'.format(text))


def to_graphviz(booster, fmap='', num_trees=0, rankdir='UT',
                yes_color='#0000FF', no_color='#FF0000', **kwargs):

    """Convert specified tree to graphviz instance. IPython can automatically plot the
    returned graphiz instance. Otherwise, you should call .render() method
    of the returned graphiz instance.

    Parameters
    ----------
    booster : Booster, XGBModel
        Booster or XGBModel instance
    fmap: str (optional)
       The name of feature map file
    num_trees : int, default 0
        Specify the ordinal number of target tree
    rankdir : str, default "UT"
        Passed to graphiz via graph_attr
    yes_color : str, default '#0000FF'
        Edge color when meets the node condition.
    no_color : str, default '#FF0000'
        Edge color when doesn't meet the node condition.
    kwargs :
        Other keywords passed to graphviz graph_attr

    Returns
    -------
    ax : matplotlib Axes
    """

    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError('You must install graphviz to plot tree')

    if not isinstance(booster, (Booster, XGBModel)):
        raise ValueError('booster must be Booster or XGBModel instance')

    if isinstance(booster, XGBModel):
        booster = booster.get_booster()

    tree = booster.get_dump(fmap=fmap)[num_trees]
    tree = tree.split()

    kwargs = kwargs.copy()
    kwargs.update({'rankdir': rankdir})
    graph = Digraph(graph_attr=kwargs)

    for i, text in enumerate(tree):
        if text[0].isdigit():
            node = _parse_node(graph, text)
        else:
            if i == 0:
                # 1st string must be node
                raise ValueError('Unable to parse given string as tree')
            _parse_edge(graph, node, text, yes_color=yes_color,
                        no_color=no_color)

    return graph


def plot_tree(booster, fmap='', num_trees=0, rankdir='UT', ax=None, **kwargs):
    """Plot specified tree.

    Parameters
    ----------
    booster : Booster, XGBModel
        Booster or XGBModel instance
    fmap: str (optional)
       The name of feature map file
    num_trees : int, default 0
        Specify the ordinal number of target tree
    rankdir : str, default "UT"
        Passed to graphiz via graph_attr
    ax : matplotlib Axes, default None
        Target axes instance. If None, new figure and axes will be created.
    kwargs :
        Other keywords passed to to_graphviz

    Returns
    -------
    ax : matplotlib Axes

    """

    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as image
    except ImportError:
        raise ImportError('You must install matplotlib to plot tree')

    if ax is None:
        _, ax = plt.subplots(1, 1)

    g = to_graphviz(booster, fmap=fmap, num_trees=num_trees, rankdir=rankdir, **kwargs)

    s = BytesIO()
    s.write(g.pipe(format='png'))
    s.seek(0)
    img = image.imread(s)

    ax.imshow(img)
    ax.axis('off')
    return ax
