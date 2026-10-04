Training times in seconds from `timer.Train.Train`, excluding DMatrix construction. Commit columns are ordered oldest to newest.

| Features | Targets | Grow policy | c2ca8c99a (s) | 05d41a90c (s) | Change vs c2ca8c99a (%) | 4f8b3185c (s) | Change vs 05d41a90c (%) |
|---------:|--------:|-------------|--------------:|--------------:|------------------------:|--------------:|------------------------:|
|      256 |       1 | depthwise   |         34.82 |         32.28 |                  -7.29% |         27.57 |                 -14.59% |
|      256 |       4 | depthwise   |        102.07 |        113.35 |                 +11.06% |         98.85 |                 -12.80% |
|      256 |       1 | lossguide   |         35.09 |         33.36 |                  -4.93% |         30.30 |                  -9.17% |
|      256 |       4 | lossguide   |        100.01 |        101.14 |                  +1.13% |        105.42 |                  +4.23% |
|      512 |       1 | depthwise   |         73.41 |         68.28 |                  -6.99% |         62.99 |                  -7.75% |
|      512 |       4 | depthwise   |        212.25 |        257.52 |                 +21.33% |        227.29 |                 -11.74% |
|      512 |       1 | lossguide   |         73.68 |         70.32 |                  -4.56% |         63.61 |                  -9.54% |
|      512 |       4 | lossguide   |        214.72 |        212.28 |                  -1.14% |        227.07 |                  +6.97% |

**Change (%)** = `(new_commit_time - previous_commit_time) / previous_commit_time × 100%`, calculated before rounding. Each change column compares the time immediately to its left with the named previous commit. Negative values mean faster training; positive values mean slower training.

All runs use an NVIDIA RTX PRO 6000 Blackwell Server Edition, 67,108,864 samples, and 128 training rounds. Four-target runs use `multi_output_tree` with `debug_synchronize=true`; single-target runs use `one_output_per_tree` with `debug_synchronize=false`.

The `c2ca8c99a` and `05d41a90c` archives report hosts with 16 CPUs and 1 GPU, driver `595.84.01`, and `dxgb_bench` version `0.1.dev407+gb237b63e0`. The `4f8b3185c` archive reports a host with 256 CPUs and 8 GPUs, driver `595.71.05`, and `dxgb_bench` version `0.1.dev399+ge42cef5c4`. All runs specify `n_workers=1`; comparisons involving `4f8b3185c` include these environment differences.

Benchmark result archives:

- Old commit `c2ca8c99a`: [rtxpro6000-c2ca8c99a.zip](../bench/rtxpro6000-c2ca8c99a.zip)
- Commit `05d41a90c`: [rtxpro6000-05d41a90c.zip](../bench/rtxpro6000-05d41a90c.zip)
- Latest commit `4f8b3185c`: [rtxpro6000-4f8b3185c.zip](../bench/rtxpro6000-4f8b3185c.zip)

Each row corresponds to the matching `incore-0.json` through `incore-7.json` files in all three archives, in that order.

Commit `4f8b3185c` is identified by its archive filename; its JSON results do not embed a `binfo.GIT_HASH`.
