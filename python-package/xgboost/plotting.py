# pylint: disable=too-many-locals, too-many-arguments, invalid-name,
# pylint: disable=too-many-branches
"""Plotting Library."""
import json
import warnings
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np

from ._typing import PathLike
from .core import Booster, _deprecate_positional_args
from .sklearn import XGBModel

Axes = Any  # real type is matplotlib.axes.Axes
GraphvizSource = Any  # real type is graphviz.Source


@_deprecate_positional_args
def plot_importance(
    booster: Union[XGBModel, Booster, dict],
    *,
    ax: Optional[Axes] = None,
    height: float = 0.2,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    title: str = "Feature importance",
    xlabel: str = "Importance score",
    ylabel: str = "Features",
    fmap: PathLike = "",
    importance_type: str = "weight",
    max_num_features: Optional[int] = None,
    grid: bool = True,
    show_values: bool = True,
    values_format: str = "{v}",
    **kwargs: Any,
) -> Axes:
    """Plot importance based on fitted trees.

    Parameters
    ----------
    booster :
        Booster or XGBModel instance, or dict taken by Booster.get_fscore()
    ax : matplotlib Axes
        Target axes instance. If None, new figure and axes will be created.
    grid :
        Turn the axes grids on or off.  Default is True (On).
    importance_type :
        How the importance is calculated: either "weight", "gain", or "cover"

        * "weight" is the number of times a feature appears in a tree
        * "gain" is the average gain of splits which use the feature
        * "cover" is the average coverage of splits which use the feature
          where coverage is defined as the number of samples affected by the split
    max_num_features :
        Maximum number of top features displayed on plot. If None, all features will be
        displayed.
    height :
        Bar height, passed to ax.barh()
    xlim :
        Tuple passed to axes.xlim()
    ylim :
        Tuple passed to axes.ylim()
    title :
        Axes title. To disable, pass None.
    xlabel :
        X axis title label. To disable, pass None.
    ylabel :
        Y axis title label. To disable, pass None.
    fmap :
        The name of feature map file.
    show_values :
        Show values on plot. To disable, pass False.
    values_format :
        Format string for values. "v" will be replaced by the value of the feature
        importance.  e.g. Pass "{v:.2f}" in order to limit the number of digits after
        the decimal point to two, for each value printed on the graph.
    kwargs :
        Other keywords passed to ax.barh()

    Returns
    -------
    ax : matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("You must install matplotlib to plot importance") from e

    if isinstance(booster, XGBModel):
        importance = booster.get_booster().get_score(
            importance_type=importance_type, fmap=fmap
        )
    elif isinstance(booster, Booster):
        importance = booster.get_score(importance_type=importance_type, fmap=fmap)
    elif isinstance(booster, dict):
        importance = booster
    else:
        raise ValueError("tree must be Booster, XGBModel or dict instance")

    if not importance:
        raise ValueError(
            "Booster.get_score() results in empty.  "
            + "This maybe caused by having all trees as decision dumps."
        )

    tuples = [(k, importance[k]) for k in importance]
    if max_num_features is not None:
        # pylint: disable=invalid-unary-operand-type
        tuples = sorted(tuples, key=lambda _x: _x[1])[-max_num_features:]
    else:
        tuples = sorted(tuples, key=lambda _x: _x[1])
    labels, values = zip(*tuples)

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ylocs = np.arange(len(values))
    ax.barh(ylocs, values, align="center", height=height, **kwargs)

    if show_values is True:
        for x, y in zip(values, ylocs):
            ax.text(x + 1, float(y), values_format.format(v=x), va="center")

    ax.set_yticks(ylocs)
    ax.set_yticklabels(labels)

    if xlim is not None:
        if not isinstance(xlim, tuple) or len(xlim) != 2:
            raise ValueError("xlim must be a tuple of 2 elements")
    else:
        xlim = (0, max(values) * 1.1)
    ax.set_xlim(xlim)

    if ylim is not None:
        if not isinstance(ylim, tuple) or len(ylim) != 2:
            raise ValueError("ylim must be a tuple of 2 elements")
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


@_deprecate_positional_args
def to_graphviz(
    booster: Union[Booster, XGBModel],
    *,
    fmap: PathLike = "",
    num_trees: Optional[int] = None,
    rankdir: Optional[str] = None,
    yes_color: Optional[str] = None,
    no_color: Optional[str] = None,
    condition_node_params: Optional[dict] = None,
    leaf_node_params: Optional[dict] = None,
    with_stats: bool = False,
    tree_idx: int = 0,
    **kwargs: Any,
) -> GraphvizSource:
    """Convert specified tree to graphviz instance. IPython can automatically plot
    the returned graphviz instance. Otherwise, you should call .render() method
    of the returned graphviz instance.

    Parameters
    ----------
    booster :
        Booster or XGBModel instance
    fmap :
       The name of feature map file
    num_trees :

        .. deprecated:: 3.0

        Specify the ordinal number of target tree

    rankdir :
        Passed to graphviz via graph_attr
    yes_color :
        Edge color when meets the node condition.
    no_color :
        Edge color when doesn't meet the node condition.
    condition_node_params :
        Condition node configuration for for graphviz.  Example:

        .. code-block:: python

            {'shape': 'box',
             'style': 'filled,rounded',
             'fillcolor': '#78bceb'}

    leaf_node_params :
        Leaf node configuration for graphviz. Example:

        .. code-block:: python

            {'shape': 'box',
             'style': 'filled',
             'fillcolor': '#e48038'}

    with_stats :

        .. versionadded:: 3.0

        Controls whether the split statistics should be included.

    tree_idx :

        .. versionadded:: 3.0

        Specify the ordinal index of target tree.

    kwargs :
        Other keywords passed to graphviz graph_attr, e.g. ``graph [ {key} = {value} ]``

    Returns
    -------
    graph: graphviz.Source

    """
    try:
        from graphviz import Source
    except ImportError as e:
        raise ImportError("You must install graphviz to plot tree") from e
    if isinstance(booster, XGBModel):
        booster = booster.get_booster()

    # squash everything back into kwargs again for compatibility
    parameters = "dot"
    extra = {}
    for key, value in kwargs.items():
        extra[key] = value

    if rankdir is not None:
        kwargs["graph_attrs"] = {}
        kwargs["graph_attrs"]["rankdir"] = rankdir
    for key, value in extra.items():
        if kwargs.get("graph_attrs", None) is not None:
            kwargs["graph_attrs"][key] = value
        else:
            kwargs["graph_attrs"] = {}
        del kwargs[key]

    if yes_color is not None or no_color is not None:
        kwargs["edge"] = {}
    if yes_color is not None:
        kwargs["edge"]["yes_color"] = yes_color
    if no_color is not None:
        kwargs["edge"]["no_color"] = no_color

    if condition_node_params is not None:
        kwargs["condition_node_params"] = condition_node_params
    if leaf_node_params is not None:
        kwargs["leaf_node_params"] = leaf_node_params

    if kwargs:
        parameters += ":"
        parameters += json.dumps(kwargs)

    if num_trees is not None:
        warnings.warn(
            "The `num_trees` parameter is deprecated, use `tree_idx` insetad. ",
            FutureWarning,
        )
        if tree_idx not in (0, num_trees):
            raise ValueError(
                "Both `num_trees` and `tree_idx` are used, prefer `tree_idx` instead."
            )
        tree_idx = num_trees

    tree = booster.get_dump(fmap=fmap, dump_format=parameters, with_stats=with_stats)[
        tree_idx
    ]
    g = Source(tree)
    return g


@_deprecate_positional_args
def plot_tree(
    booster: Union[Booster, XGBModel],
    *,
    fmap: PathLike = "",
    num_trees: Optional[int] = None,
    rankdir: Optional[str] = None,
    ax: Optional[Axes] = None,
    with_stats: bool = False,
    tree_idx: int = 0,
    **kwargs: Any,
) -> Axes:
    """Plot specified tree.

    Parameters
    ----------
    booster :
        Booster or XGBModel instance
    fmap: str (optional)
       The name of feature map file
    num_trees :

        .. deprecated:: 3.0

    rankdir : str, default "TB"
        Passed to graphviz via graph_attr
    ax : matplotlib Axes, default None
        Target axes instance. If None, new figure and axes will be created.

    with_stats :

        .. versionadded:: 3.0

        See :py:func:`to_graphviz`.

    tree_idx :

        .. versionadded:: 3.0

        See :py:func:`to_graphviz`.

    kwargs :
        Other keywords passed to :py:func:`to_graphviz`

    Returns
    -------
    ax : matplotlib Axes

    """
    try:
        from matplotlib import image
        from matplotlib import pyplot as plt
    except ImportError as e:
        raise ImportError("You must install matplotlib to plot tree") from e

    if ax is None:
        _, ax = plt.subplots(1, 1)

    g = to_graphviz(
        booster,
        fmap=fmap,
        num_trees=num_trees,
        rankdir=rankdir,
        with_stats=with_stats,
        tree_idx=tree_idx,
        **kwargs,
    )

    s = BytesIO()
    s.write(g.pipe(format="png"))
    s.seek(0)
    img = image.imread(s)

    ax.imshow(img)
    ax.axis("off")
    return ax
