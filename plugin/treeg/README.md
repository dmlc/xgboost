# Tree-G: Graph-Aware Decision Trees

This plugin adds a `treeg` booster to XGBoost for **vertex prediction on graphs**. It
implements Tree-G ([Bechler-Speicher et al., AAAI 2024](https://arxiv.org/abs/2207.02760)).

Standard gradient-boosted trees split on one feature of one sample, `x[i][k] > θ`. On graph
data, a vertex's label often depends on its neighbourhood as much as on its own features.
The usual workaround is to engineer aggregate features before training, which fixes the
aggregation in advance. Tree-G makes the aggregation part of the split search, so each
split *learns* which neighbourhood matters. The model stays a gradient-boosted tree
ensemble, with the same objectives, regularization and early stopping.

## How it works

At vertex `i`, a Tree-G split compares a **propagated feature** to a threshold:

$$
\phi(i) = \big( (A^d \circ M_r(S))\, \mathbf{f}_k \big)_i \;>\; \theta
$$

The split search chooses:

| Symbol | Meaning |
| --- | --- |
| `k` | A node feature (column of `X`). |
| `d` | Walk length, `0 ≤ d ≤ max_walk_len`. $(A^d \mathbf{f}_k)_i$ sums feature `k` over all length-`d` walks ending at `i`. `d = 0` is an ordinary split on `X[i, k]`. |
| `r` | Walk restriction to a vertex subset `S`. **Source**: only walks that *start* in `S`. **Cycle**: only walks that start and end at the same vertex of `S`. |
| `S` | The **used subset**, chosen by pointing at an ancestor split node (see below). |
| `θ` | Threshold. |

**Subsets and the pointer mechanism.** Every split node `u` also partitions its own used
subset into two generated subsets: `S_{u,+}` (vertices whose $\phi > \theta_u$) and
`S_{u,−}` (the rest). A descendant split can then use one of these as its `S`. It
"points" at an ancestor at most `max_ancestor_distance` levels up and picks `+` or `−`.
It can instead use all vertices `V`, which is always the root's choice. Through this
mechanism, a deeper split can ask, for example, "the sum of feature 3 over neighbours that
an earlier split found to have high feature 1". The subsets are recomputed for each graph,
so they are not stored in the model.

Setting `max_ancestor_distance = 0` disables subsets (always `S = V`). Setting
`max_walk_len = 0` additionally ignores the graph, which reduces Tree-G to a standard
tree on `X`. The paper's ablations show that both mechanisms contribute. It reports that
`d ≤ 2` and `a ≤ 2` are sufficient, which are the defaults here.

**Transductive setting.** The whole graph, including test vertices, is present during
training. Test vertices get weight `0`, so they never affect the loss, but their features
still flow into their neighbours' aggregates.

## Build

The plugin is off by default. From the `xgboost` directory:

```bash
cmake -B build -S . -DPLUGIN_TREEG=ON
cmake --build build -j
pip install ./python-package
```

## Quick start

```python
import numpy as np
import scipy.sparse as sp
import xgboost as xgb

rng = np.random.default_rng(0)
n, k, d = 200, 4, 2

# A random undirected graph, node features, and a label that depends on the neighbours.
A = sp.random(n, n, density=0.02, random_state=0, format="csr")
A = ((A + A.T) > 0).astype(np.float32)
X = rng.normal(size=(n, k)).astype(np.float32)
y = A @ X[:, 0]                       # sum of feature 0 over each vertex's neighbours

# Input layout: [X | A | A^2]
blocks = [sp.csr_matrix(X)]
A_pow = sp.identity(n, dtype=np.float32, format="csr")
for _ in range(d):
    A_pow = A_pow @ A
    blocks.append(A_pow)
data = sp.hstack(blocks, format="csr")   # shape (n, k + n*d)

# Train/test split by weight: test vertices stay in the graph with weight 0.
train = rng.random(n) < 0.7
dtrain = xgb.DMatrix(data, label=y, weight=train.astype(np.float32))

params = {
    "booster": "treeg",
    "objective": "reg:squarederror",
    "num_vertices": n,    # required
    "num_features": k,    # required
    "max_walk_len": d,    # must match the number of adjacency blocks
    "max_leaves": 8,
}
model = xgb.train(params, dtrain, num_boost_round=50)

pred = model.predict(dtrain)             # one prediction per vertex
test = ~train
print("test RMSE:", np.sqrt(np.mean((pred[test] - y[test]) ** 2)))
```


## Input format

One `DMatrix` row is one vertex, laid out as:

```
[ X | A | A² | ... | A^d ]
```

| Block | Shape | Meaning |
| --- | --- | --- |
| `X` | `n × k` | Node features. |
| `A^i` | `n × n` | `i`-th power of the adjacency matrix, row-flattened. `A[i, j]` is the edge weight from `j` to `i`. |

* `d` must equal `max_walk_len`.
* **Use sparse input** (`scipy.sparse.csr_matrix`) for anything but tiny graphs. A dense
  input stores all $n^2 d$ entries.
* **Labels**: one per vertex. Use any value for test vertices.
* **Weights**: `1` for training vertices, `0` for held-out vertices.
* Directed and weighted graphs are supported. For walks in the opposite direction, pass
  powers of $A^\top$.

## Parameters

### Graph

| Parameter | Default | Description |
| --- | --- | --- |
| `num_vertices` | **required** | Number of vertices `n`. Must equal the number of rows. |
| `num_features` | **required** | Number of node features `k`. |
| `max_walk_len` | `2` | Maximum walk length `d`. |
| `max_ancestor_distance` | `2` | How many levels up a split may look for a subset. `0` disables subsets. |
| `intercept` | `false` | Adds a constant-1 feature, so $(A^d \mathbf{1})_i$ counts length-`d` walks, for example the vertex degree when `d = 1`. Has no effect when `max_walk_len = 0`. |
| `attention_type_sample_probability` | `1.0` | Per node, include each restriction type (source, cycle) with this probability. Values `< 1` add randomness and speed up training. Not saved with the model. |

### Tree growth and regularization

These behave exactly as in `gbtree`.

| Parameter | Default | Description |
| --- | --- | --- |
| `learning_rate` | `0.3` | Step size shrinkage. |
| `max_leaves` | `32` | Maximum leaves per tree. Trees grow leaf-wise (best-first); there is no `max_depth`. |
| `min_split_loss` | `0.0` | Minimum loss reduction required to split. |
| `min_child_weight` | `1.0` | Minimum sum of hessian in a child. |
| `reg_lambda` | `1.0` | L2 regularization on leaf weights. |
| `reg_alpha` | `0.0` | L1 regularization on leaf weights. |
| `max_delta_step` | `0.0` | Maximum delta step for leaf weights. |
| `colsample_bytree` | `1.0` | Fraction of node features sampled per tree. |

### Performance

| Parameter | Default | Description |
| --- | --- | --- |
| `nthread` | all cores | Results are bit-identical for any thread count. |
| `max_cached_split_bytes` | 1 GiB | Memory for caching split values that are reused across a tree. Lower it to reduce peak memory, or set it to `0` to disable the cache. Either makes training slower. An `INFO` log reports how much of the cache was used. |

## Memory and scaling

* Memory is dominated by the adjacency powers, which are stored sparsely. Dense graphs and large `d` grow quickly, because `A^d` fills in.
* On top of that, training uses up to `max_cached_split_bytes`.
* Training time is roughly linear in the number of features, `max_walk_len`, and `max_ancestor_distance`, and linear in the number of boosting rounds.
* The graph must fit in memory, with `n < 2^31` and `k + n·d < 2^32`.

## Supported features

* Regression and binary classification objectives.
* Multi-class classification, one-vs-all (one tree per class per round).
* `save_model` / `load_model`.
* Early stopping and `predict(..., iteration_range=(a, b))`.
* `get_dump()` (text format). A split prints as `[f<k> <= θ]`, and vertices with value `<= θ` go left.
* `get_score()` with `weight`, `gain`, `total_gain`, `cover`, `total_cover`. Each
  feature's score is summed over all walk lengths. Splits on the intercept are not counted.

## Limitations

These raise `XGBoostError`:

* GPU (`device="cuda"`). Tree-G runs on CPU only.
* Distributed training (Dask, Spark) and external-memory `DMatrix`.
* `multi_strategy="multi_output_tree"`, `process_type="update"`.
* `predict(pred_leaf=True)`, `pred_contribs=True`, `pred_interactions=True` (no SHAP).
* Model slicing (`model[a:b]`). Use `iteration_range` instead.
* `get_dump(dump_format="json" | "dot")` and `trees_to_dataframe()`.
* Categorical features.
* Graph-level prediction (paper §3.5). Only vertex prediction is implemented.

Accepted but with no effect:

* `max_depth` is ignored with XGBoost's standard warning. Use `max_leaves`.
* **Missing values** are treated as `0.0`, including `NaN` in `X`. There is no learned
  default direction. If zero is not meaningful for a feature, impute it yourself first.

## Citation

```bibtex
@inproceedings{bechler2024treeg,
  title     = {TREE-G: Decision Trees Contesting Graph Neural Networks},
  author    = {Bechler-Speicher, Maya and Globerson, Amir and Gilad-Bachrach, Ran},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2024}
}
```

## Contributing

Implementation notes (architecture, split search, memory layout, design decisions) are in
[DESIGN.md](DESIGN.md).

```bash
# C++ tests
cmake -B build -S . -DPLUGIN_TREEG=ON -DGOOGLE_TEST=ON
cmake --build build -j
./build/testxgboost --gtest_filter='*TreeG*'

# Python tests
pytest tests/python/test_treeg.py
```

CI runs both via [`ops/pipeline/build-test-treeg.sh`](../../ops/pipeline/build-test-treeg.sh).
