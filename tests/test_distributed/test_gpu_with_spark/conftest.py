from typing import Sequence

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: Sequence) -> None:
    # mark dask tests as `mgpu`.
    mgpu_mark = pytest.mark.mgpu
    for item in items:
        item.add_marker(mgpu_mark)
