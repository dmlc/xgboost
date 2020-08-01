###########
Tree Splits
###########

*************
Trivial Split
*************

This note applies only to ``hist`` and ``gpu_hist`` tree methods.  Trivial split means
during a tree split, all valid data goes to one side of the tree, while missing value goes
to other side.  In XGBoost, we defined the constant threshold `kTrivialSplit` as ``-inf``.
