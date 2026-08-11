import pytest
from xgboost.testing.monotone_constraints import (
    run_monotone_constraints,
    run_multi_output_monotone,
    run_parent_gain,
)


@pytest.mark.parametrize(
    "tree_method,policy",
    [
        ("hist", "depthwise"),
        ("approx", "depthwise"),
        ("hist", "lossguide"),
        ("approx", "lossguide"),
    ],
)
def test_gpu_monotone_constraints(tree_method: str, policy: str) -> None:
    run_monotone_constraints("cuda", tree_method, policy)


@pytest.mark.parametrize("multi_strategy", ["one_output_per_tree", "multi_output_tree"])
def test_parent_gain(multi_strategy: str) -> None:
    run_parent_gain("cuda", multi_strategy)


@pytest.mark.parametrize("policy", ["depthwise", "lossguide"])
def test_vector_leaf_monotone(policy: str) -> None:
    run_monotone_constraints("cuda", "hist", policy, multi_strategy="multi_output_tree")


@pytest.mark.parametrize("multi_strategy", ["one_output_per_tree", "multi_output_tree"])
@pytest.mark.parametrize("policy", ["depthwise", "lossguide"])
def test_deep_monotone(policy: str, multi_strategy: str) -> None:
    run_multi_output_monotone("cuda", policy, multi_strategy)
