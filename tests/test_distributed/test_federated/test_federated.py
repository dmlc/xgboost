import pytest

from xgboost.testing.federated import run_federated_learning


@pytest.mark.parametrize("with_ssl", [True, False])
def test_federated_learning(with_ssl: bool) -> None:
    run_federated_learning(with_ssl, False, __file__)
