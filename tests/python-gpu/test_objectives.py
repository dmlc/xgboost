from xgboost import testing as tm
from xgboost.core import ExtMemQuantileDMatrix


def test_get_info_batches() -> None:
    n_batches = 3
    it = tm.IteratorForTest(
        *tm.make_batches(16, 4, n_batches, use_cupy=True),
        cache="cache",
        on_host=True,
        min_cache_page_bytes=0
    )
    Xy = ExtMemQuantileDMatrix(it)
    k = 0
    for p in Xy.iter_info_batches():
        k += 1
    assert k == n_batches
