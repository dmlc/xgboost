"""Experimental support for a new objective interface with target dimension
reduction.

.. warning::

  Do not use this module unless you want to participate in development.

.. versionadded:: 3.2.0

"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple

from ._typing import ArrayLike

if TYPE_CHECKING:
    from .core import DMatrix


class Objective(ABC):
    """Base class for custom objective function.

    .. warning::

        Do not use this class unless you want to participate in development.

    """

    @abstractmethod
    def __call__(
        self, y_pred: ArrayLike, dtrain: "DMatrix"
    ) -> Tuple[ArrayLike, ArrayLike]: ...


class TreeObjective(Objective):
    """Base class for tree-specific custom objective function.

    .. warning::

        Do not use this class unless you want to participate in development.

    """

    def split_grad(
        self, grad: ArrayLike, hess: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Provide different gradient type for finding tree structure."""
        return grad, hess
