Training times in seconds from `timer.Train.Train`, excluding DMatrix construction. Commit columns are ordered oldest to newest.

| Features | Targets | Grow policy | c2ca8c99a (old, s) | 05d41a90c (new, s) | Change (%) |
|---------:|--------:|-------------|-------------------:|-------------------:|-----------:|
|      256 |       1 | depthwise   |              34.82 |              32.28 |     -7.29% |
|      256 |       4 | depthwise   |             102.07 |             113.35 |    +11.06% |
|      256 |       1 | lossguide   |              35.09 |              33.36 |     -4.93% |
|      256 |       4 | lossguide   |             100.01 |             101.14 |     +1.13% |
|      512 |       1 | depthwise   |              73.41 |              68.28 |     -6.99% |
|      512 |       4 | depthwise   |             212.25 |             257.52 |    +21.33% |
|      512 |       1 | lossguide   |              73.68 |              70.32 |     -4.56% |
|      512 |       4 | lossguide   |             214.72 |             212.28 |     -1.14% |

**Change (%)** = `(new_commit_time - old_commit_time) / old_commit_time × 100%`, calculated before rounding. Negative values mean faster training; positive values mean slower training.

All runs use an NVIDIA RTX PRO 6000 Blackwell Server Edition, 67,108,864 samples, and 128 training rounds. Four-target runs use `multi_output_tree` with `debug_synchronize=true`; single-target runs use `one_output_per_tree` with `debug_synchronize=false`.

Benchmark result archives:

- Old commit `c2ca8c99a`: [rtxpro6000-c2ca8c99a.zip](../bench/rtxpro6000-c2ca8c99a.zip)
- New commit `05d41a90c`: [rtxpro6000-05d41a90c.zip](../bench/rtxpro6000-05d41a90c.zip)

Each row corresponds to the matching `incore-0.json` through `incore-7.json` files in the two archives, in that order.