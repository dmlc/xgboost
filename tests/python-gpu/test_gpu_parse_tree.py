import sys

sys.path.append("tests/python")
from test_parse_tree import TestTreesToDataFrame


def test_tree_to_df_categorical():
    cputest = TestTreesToDataFrame()
    cputest.run_tree_to_df_categorical("gpu_hist")


def test_split_value_histograms():
    cputest = TestTreesToDataFrame()
    cputest.run_split_value_histograms("gpu_hist")
