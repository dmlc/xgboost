from xgboost.testing.multi_target import run_multiclass, run_multilabel


def test_multiclass() -> None:
    run_multiclass("cuda", 1.0)


def test_multilabel() -> None:
    run_multilabel("cuda", 1.0)
