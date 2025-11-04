from xgboost.testing.multi_target import run_multiclass, run_multilabel


def test_multiclass() -> None:
    # learning_rate is not yet supported.
    run_multiclass("cuda", 1.0)


def test_multilabel() -> None:
    # learning_rate is not yet supported.
    run_multilabel("cuda", 1.0)
