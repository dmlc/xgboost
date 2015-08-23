# coding: utf-8
# pylint: disable=too-many-locals, too-many-arguments, invalid-name,
# pylint: disable=too-many-branches
"""Plotting Library."""
from __future__ import absolute_import

import re
import numpy as np
from .core import Booster

from io import BytesIO

def plot_importance(booster, ax=None, height=0.2,
                    xlim=None, title='Feature importance',
                    xlabel='F score', ylabel='Features',
                    grid=True, **kwargs):

    """Plot importance based on fitted trees.

    Parameters
    ----------
    booster : Booster or dict
        Booster instance, or dict taken by Booster.get_fscore()
    ax : matplotlib Axes, default None
        Target axes instance. If None, new figure and axes will be created.
    height : float, default 0.2
        Bar height, passed to ax.barh()
    xlim : tuple, default None
        Tuple passed to axes.xlim()
    title : str, default "Feature importance"
        Axes title. To disable, pass None.
    xlabel : str, default "F score"
        X axis title label. To disable, pass None.
    ylabel : str, default "Features"
        Y axis title label. To disable, pass None.
    kwargs :
        Other keywords passed to ax.barh()

    Returns
    -------
    ax : matplotlib Axes
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('You must install matplotlib to plot importance')

    if isinstance(booster, Booster):
        importance = booster.get_fscore()
    elif isinstance(booster, dict):
        importance = booster
    else:
        raise ValueError('tree must be Booster or dict instance')

    if len(importance) == 0:
        raise ValueError('Booster.get_fscore() results in empty')

    tuples = [(k, importance[k]) for k in importance]
    tuples = sorted(tuples, key=lambda x: x[1])
    labels, values = zip(*tuples)

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ylocs = np.arange(len(values))
    ax.barh(ylocs, values, align='center', height=height, **kwargs)

    for x, y in zip(values, ylocs):
        ax.text(x + 1, y, x, va='center')

    ax.set_yticks(ylocs)
    ax.set_yticklabels(labels)

    if xlim is not None:
        if not isinstance(xlim, tuple) or len(xlim, 2):
            raise ValueError('xlim must be a tuple of 2 elements')
    else:
        xlim = (0, max(values) * 1.1)
    ax.set_xlim(xlim)

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
    raise ValueError('Unable to parse edge: {0}'.format(text))


def to_graphviz(booster, num_trees=0, rankdir='UT',
                yes_color='#0000FF', no_color='#FF0000', **kwargs):

    """Convert specified tree to graphviz instance. IPython can automatically plot the
    returned graphiz instance. Otherwise, you shoud call .render() method
    of the returned graphiz instance.

    Parameters
    ----------
    booster : Booster
        Booster instance
    num_trees : int, default 0
        Specify the ordinal number of target tree
    rankdir : str, default "UT"
        Passed to graphiz via graph_attr
    yes_color : str, default '#0000FF'
        Edge color when meets the node condigion.
    no_color : str, default '#FF0000'
        Edge color when doesn't meet the node condigion.
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

    if not isinstance(booster, Booster):
        raise ValueError('booster must be Booster instance')

    tree = booster.get_dump()[num_trees]
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


def plot_tree(booster, num_trees=0, rankdir='UT', ax=None, **kwargs):
    """Plot specified tree.

    Parameters
    ----------
    booster : Booster
        Booster instance
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

    g = to_graphviz(booster, num_trees=num_trees, rankdir=rankdir, **kwargs)

    s = BytesIO()
    s.write(g.pipe(format='png'))
    s.seek(0)
    img = image.imread(s)

    ax.imshow(img)
    ax.axis('off')
    return ax
