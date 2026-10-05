Training times in seconds from `timer.Train.Train`, excluding DMatrix construction. Commit columns are ordered oldest to newest.

**RTX PRO 6000 Blackwell**

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

RTX PRO 6000 runs use an NVIDIA RTX PRO 6000 Blackwell Server Edition, 67,108,864 samples, and 128 training rounds. Four-target runs use `multi_output_tree` with `debug_synchronize=true`; single-target runs use `one_output_per_tree` with `debug_synchronize=false`.

The `c2ca8c99a` and `05d41a90c` archives report hosts with 16 CPUs and 1 GPU, driver `595.84.01`, and `dxgb_bench` version `0.1.dev407+gb237b63e0`. The `4f8b3185c` archive reports a host with 256 CPUs and 8 GPUs, driver `595.71.05`, and `dxgb_bench` version `0.1.dev399+ge42cef5c4`. All runs specify `n_workers=1`; comparisons involving `4f8b3185c` include these environment differences.

Benchmark result archives:

- Old commit `c2ca8c99a`: [rtxpro6000-c2ca8c99a.zip](../bench/rtxpro6000-c2ca8c99a.zip)
- Commit `05d41a90c`: [rtxpro6000-05d41a90c.zip](../bench/rtxpro6000-05d41a90c.zip)
- Latest commit `4f8b3185c`: [rtxpro6000-4f8b3185c.zip](../bench/rtxpro6000-4f8b3185c.zip)

Each row corresponds to the matching `incore-0.json` through `incore-7.json` files in all three archives, in that order.

Commit `4f8b3185c` is identified by its archive filename; its JSON results do not embed a `binfo.GIT_HASH`.

**GH200 (NVIDIA GH200 480GB)**

| Features | Targets | Grow policy | c2ca8c99a (s) | 4f8b3185c (s) | Change vs c2ca8c99a (%) |
|---------:|--------:|-------------|--------------:|--------------:|------------------------:|
|      256 |       1 | depthwise   |         40.45 |         39.95 |                  -1.24% |
|      256 |       4 | depthwise   |        149.95 |        150.91 |                  +0.64% |
|      256 |       1 | lossguide   |         40.64 |         40.54 |                  -0.25% |
|      256 |       4 | lossguide   |        151.54 |        153.55 |                  +1.33% |
|      512 |       1 | depthwise   |         85.98 |         85.70 |                  -0.33% |
|      512 |       4 | depthwise   |        330.37 |        333.68 |                  +1.00% |
|      512 |       1 | lossguide   |         86.21 |         86.42 |                  +0.24% |
|      512 |       4 | lossguide   |        332.01 |        335.51 |                  +1.05% |

Training times use `timer.Train.Train` and exclude DMatrix construction. Change is `(4f8b3185c_time - c2ca8c99a_time) / c2ca8c99a_time × 100%`, calculated before rounding; negative values mean faster training.

Both archives report the same environment: Linux `aarch64`, 72 CPUs, 1 NVIDIA GH200 480GB GPU, driver `580.95.05`, CUDA 13.3, and `dxgb_bench` version `0.1.dev397+gb8d464d03`. All runs use 67,108,864 samples, 128 training rounds, and `n_workers=1`. Four-target runs use `multi_output_tree` with `debug_synchronize=true`; single-target runs use `one_output_per_tree` with `debug_synchronize=false`.

Benchmark result archives:

- Old commit `c2ca8c99a`: [gh200-c2ca8c99a.zip](../bench/gh200-c2ca8c99a.zip)
- Latest commit `4f8b3185c`: [gh200-4f8b3185c.zip](../bench/gh200-4f8b3185c.zip)

Each row corresponds to the matching `incore-0.json` through `incore-7.json` files in the two GH200 archives, in that order. Commit labels come from the archive filenames; the JSON results do not embed a `binfo.GIT_HASH`.

**RTX PRO 6000 Blackwell (imbalanced histograms)**

| Features | Targets | Max depth | Grow policy | c2ca8c99a (s) | 4f8b3185c (s) | Change vs c2ca8c99a (%) |
|---------:|--------:|----------:|-------------|--------------:|--------------:|------------------------:|
|     4080 |       1 |         6 | depthwise   |         27.80 |         25.35 |                  -8.79% |
|     4080 |       1 |         8 | depthwise   |         40.65 |         34.07 |                 -16.20% |
|     4080 |       1 |         6 | lossguide   |         30.67 |         28.62 |                  -6.69% |
|     4080 |       1 |         8 | lossguide   |         44.17 |         39.68 |                 -10.18% |
|     4080 |       4 |         6 | depthwise   |        111.08 |         96.81 |                 -12.85% |
|     4080 |       4 |         8 | depthwise   |        170.33 |        126.26 |                 -25.87% |
|     4080 |       4 |         6 | lossguide   |        110.05 |         98.46 |                 -10.53% |
|     4080 |       4 |         8 | lossguide   |        173.67 |        132.08 |                 -23.95% |

Training times use `timer.Train.Train` and exclude DMatrix construction. Change is `(4f8b3185c_time - c2ca8c99a_time) / c2ca8c99a_time × 100%`, calculated before rounding; negative values mean faster training.

All imbalanced-histogram runs use 4,194,304 samples, 4,080 features including 3,072 binary features (`n_binary=3072`), 128 training rounds, `max_bin=256`, and `data_seed=2026`. Four-target runs use `multi_output_tree` with `debug_synchronize=true`; single-target runs use `one_output_per_tree` with `debug_synchronize=false`.

Both archives report the same environment: Linux `x86_64`, a host with 256 CPUs and 8 NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs, driver `595.71.05`, CUDA 13.3, and `dxgb_bench` version `0.1.dev399+ge42cef5c4`.

Benchmark result archives:

- Old commit `c2ca8c99a`: [rtxpro6000-imb-c2ca8c99a.zip](../bench/rtxpro6000-imb-c2ca8c99a.zip)
- Latest commit `4f8b3185c`: [rtxpro6000-imb-4f8b3185c.zip](../bench/rtxpro6000-imb-4f8b3185c.zip)

Each row corresponds to the matching `incore-0.json` through `incore-7.json` files in the two imbalanced-histogram archives, in that order. Commit labels come from the archive filenames; the JSON results do not embed a `binfo.GIT_HASH`.

**RTX 4070 Ti SUPER**

| Features | Targets | Grow policy | c2ca8c99a (s) | 4161c075a (s) | Change vs c2ca8c99a (%) |
|---------:|--------:|-------------|--------------:|--------------:|------------------------:|
|      256 |       1 | depthwise   |         21.95 |         17.82 | -18.81%                 |
|      256 |       4 | depthwise   |         70.84 |         62.67 | -11.54%                 |
|      256 |       1 | lossguide   |         22.48 |         18.59 | -17.33%                 |
|      256 |       4 | lossguide   |         73.25 |         64.88 | -11.44%                 |
|      512 |       1 | depthwise   |         46.84 |         37.64 | -19.63%                 |
|      512 |       4 | depthwise   |        160.94 |        143.38 | -10.91%                 |
|      512 |       1 | lossguide   |         47.59 |         37.88 | -20.41%                 |
|      512 |       4 | lossguide   |        164.33 |        142.12 | -13.52%                 |

Training times use `timer.Train.Train` and exclude DMatrix construction and data generation. Change is `(4161c075a_time - c2ca8c99a_time) / c2ca8c99a_time × 100%`, calculated before rounding; negative values mean faster training. Commit columns are ordered oldest to newest.

Both archives report the same environment: Linux `x86_64`, 24 CPUs, 1 NVIDIA GeForce RTX 4070 Ti SUPER GPU, driver `595.45.04`, CUDA 13.3, and `dxgb_bench` version `0.1.dev396+g93fd73112.d20260924`. Recorded parameters, machine metadata, versions, and build information match for every paired case.

All runs use `qdm-iter` with 16 batches of 2^20 samples (16,777,216 samples), 128 training rounds, maximum depth 6, `max_bin=256`, and `n_workers=1`. Four-target runs use `multi_output_tree` with `debug_synchronize=true`; single-target runs use `one_output_per_tree` with `debug_synchronize=false`.

Benchmark result archives:

- Old commit `c2ca8c99a`: [4070tis-c2ca8c99a.zip](/home/jiamingy/ws/xgboost_dev/bench/4070tis/4070tis-c2ca8c99a.zip)
- New commit `4161c075a`: [4070tis-4161c075a.zip](/home/jiamingy/ws/xgboost_dev/bench/4070tis/4070tis-4161c075a.zip)

Cases are paired by feature count, grow policy, and target configuration. Commit labels come from the archive filenames; the JSON results do not embed a `binfo.GIT_HASH`.

Single-target training time decreases in all four cases (17.33–20.41%). Vector-leaf training time decreases in all four cases (10.91–13.52%). Training RMSE matches exactly at the precision stored in the JSON files at every boosting round for all eight pairs. Each commit has one recorded timing per case, so these results do not measure run-to-run variation.

**DGX Spark (NVIDIA GB10)**

| Features | Targets | Grow policy | c2ca8c99a (s) | 4161c075a (s) | Change vs c2ca8c99a (%) |
|---------:|--------:|-------------|--------------:|--------------:|------------------------:|
|      256 |       1 | depthwise   |         48.56 |         33.15 | -31.74%                 |
|      256 |       4 | depthwise   |        153.72 |         98.91 | -35.66%                 |
|      256 |       1 | lossguide   |         48.01 |         33.55 | -30.11%                 |
|      256 |       4 | lossguide   |        155.49 |        100.59 | -35.30%                 |
|      512 |       1 | depthwise   |        105.43 |         66.56 | -36.87%                 |
|      512 |       4 | depthwise   |        392.73 |        228.09 | -41.92%                 |
|      512 |       1 | lossguide   |        105.80 |         65.26 | -38.31%                 |
|      512 |       4 | lossguide   |        535.14 |        215.40 | -59.75%                 |

Training times use `timer.Train.Train` and exclude DMatrix construction and data generation. Change is `(4161c075a_time - c2ca8c99a_time) / c2ca8c99a_time × 100%`, calculated before rounding; negative values mean faster training. Commit columns are ordered oldest to newest.

Both archives report the same environment: Linux `aarch64`, 20 CPUs, 1 NVIDIA GB10 GPU, driver `580.173.02`, CUDA 13.3, and `dxgb_bench` version `0.1.dev397+gb8d464d03`. Recorded parameters, machine metadata, versions, and build information match for every paired case.

All runs use `qdm-iter` with 16 batches of 2^20 samples (16,777,216 samples), 128 training rounds, maximum depth 6, `max_bin=256`, and `n_workers=1`. Four-target runs use `multi_output_tree` with `debug_synchronize=true`; single-target runs use `one_output_per_tree` with `debug_synchronize=false`.

Benchmark result archives:

- Old commit `c2ca8c99a`: [spark-c2ca8c99a.zip](/home/jiamingy/ws/xgboost_dev/bench/DGX_Spark/spark-c2ca8c99a.zip)
- New commit `4161c075a`: [spark-4161c075a.zip](/home/jiamingy/ws/xgboost_dev/bench/DGX_Spark/spark-4161c075a.zip)

Cases are paired by feature count, grow policy, and target configuration. Commit labels come from the archive filenames; the JSON results do not embed a `binfo.GIT_HASH`.

Single-target training time decreases in all four cases (30.11–38.31%). Vector-leaf training time decreases in all four cases (35.30–59.75%). Training RMSE matches exactly at the precision stored in the JSON files at every boosting round for all eight pairs. Each commit has one recorded timing per case, so these results do not measure run-to-run variation.

**A100 (NVIDIA A100 80GB PCIe)**

| Features | Targets | Grow policy | c2ca8c99a (s) | 4161c075a (s) | Change vs c2ca8c99a (%) |
|---------:|--------:|-------------|--------------:|--------------:|------------------------:|
|      256 |       1 | depthwise   |         78.19 |         59.17 |                 -24.32% |
|      256 |       4 | depthwise   |        230.79 |        228.26 |                  -1.09% |
|      256 |       1 | lossguide   |         78.36 |         59.73 |                 -23.78% |
|      256 |       4 | lossguide   |        231.46 |        223.94 |                  -3.25% |
|      512 |       1 | depthwise   |        168.09 |        130.28 |                 -22.49% |
|      512 |       4 | depthwise   |        508.59 |        528.14 |                  +3.85% |
|      512 |       1 | lossguide   |        168.35 |        127.61 |                 -24.20% |
|      512 |       4 | lossguide   |        509.36 |        505.11 |                  -0.83% |

Training times use `timer.Train.Train`, excluding DMatrix construction and data generation. Changes are `100 × (time / reference_time − 1)`, calculated before rounding; each column names its reference. Negative values mean less training time. Commit columns are ordered oldest to newest.

Both archives report the same environment: Linux `x86_64`, a host with 32 CPUs and 2 NVIDIA A100 80GB PCIe GPUs, driver `580.159.04`, CUDA 13.2, and `dxgb_bench` version `0.1.dev410+g97feb1fd5`. Recorded parameters and machine metadata match for every paired case. Build information and versions match except for `binfo.GIT_HASH` and the XGBoost version suffix, which identify `c2ca8c99a` and `4161c075a`, respectively.

All runs use `qdm-iter` with 64 batches of 2^20 samples (67,108,864 samples), 128 training rounds, maximum depth 6, `max_bin=256`, and `n_workers=1`. Four-target runs use `multi_output_tree` with `debug_synchronize=true`; single-target runs use `one_output_per_tree` with `debug_synchronize=false`.

Training RMSE matches exactly at every recorded round for 8 of 8 case/revision pairs against `c2ca8c99a`. 0 pairs differ; 0 pairs have missing or incomplete RMSE histories. Each revision has one timing per case; run-to-run variation is not measured.

Benchmark result archives (commit labels confirmed by `binfo.GIT_HASH`):

- `c2ca8c99a`: [a100-c2ca8c99a.zip](</home/jiamingy/ws/xgboost_dev/bench/A100/a100-c2ca8c99a.zip>)
- `4161c075a`: [a100-4161c075a.zip](</home/jiamingy/ws/xgboost_dev/bench/A100/a100-4161c075a.zip>)

Single-target training time decreases in all four cases (22.49–24.32%). Vector-leaf training time decreases by 0.83–3.25% in three cases; the 512-feature `depthwise` case increases by 3.85%.
