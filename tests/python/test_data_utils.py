import gc
from typing import Literal, cast

import numpy as np
import pytest

from xgboost._data_utils import ArrayInf, from_array_interface


@pytest.mark.parametrize("zero_copy", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "empty"])
def test_array_interface_cyclic_garbage(
    zero_copy: bool, layout: Literal["contiguous", "strided", "empty"]
) -> None:
    data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    if layout == "strided":
        data = data[:, ::2]
    elif layout == "empty":
        data = data[:0]

    interface = cast(ArrayInf, data.__array_interface__)
    np.testing.assert_array_equal(
        from_array_interface(interface, zero_copy=zero_copy), data
    )
    gc.collect()
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(10):
            from_array_interface(interface, zero_copy=zero_copy)
        assert gc.collect() == 0
    finally:
        gc.collect()
        if gc_enabled:
            gc.enable()
