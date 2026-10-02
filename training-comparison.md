# Training comparison: master vs. optimize-hist

Source: [sep-28-ha.org](/home/jiamingy/ws/xgboost_dev/bench/sep-28-ha.org).

For the first table, the master baseline is commit `c2ca8c99a` ("Fix and unify base weight handling. (#12602)"). The source identifies `optimize-hist` as its branch head without recording a commit hash.

Training times below use only `Train Train ended in` (the nested `Train["Train"]` value), excluding DMatrix construction and data generation. All recorded runs trained for 128 iterations. Times are rounded to three decimal places; percentages are calculated from the full recorded precision.

Training-time change = `100 × (optimize-hist / master − 1)`. A negative value is a decrease in time (faster); a positive value is an increase in time (slower).

The commands shown for A100 and H200 specify 64 batches of 2^20 samples (67,108,864 samples); RTX 4070tis specifies 16 batches (16,777,216 samples). DGX Spark has no sample-count command recorded. The source notes that sample counts vary due to memory constraints. Feature counts and grow policies below follow the case headings. The source also specifies `use_cuda_async_pool=true`, with other parameters at their defaults.

| GPU (architecture)  | Features | Grow policy | Master training (s) | optimize-hist training (s) | Training-time change vs. master |
|---------------------|---------:|-------------|--------------------:|---------------------------:|---------------------------------|
| A100 (sm_80)        |      256 | depthwise   |              78.570 |                     64.726 | -17.62%                         |
| A100 (sm_80)        |      256 | lossguide   |              78.896 |                     65.389 | -17.12%                         |
| A100 (sm_80)        |      512 | depthwise   |             169.553 |                    142.355 | -16.04%                         |
| A100 (sm_80)        |      512 | lossguide   |             169.766 |                    141.096 | -16.89%                         |
| H200 (sm_90a)       |      256 | depthwise   |              40.415 |                     40.015 | -0.99%                          |
| H200 (sm_90a)       |      256 | lossguide   |              40.619 |                     41.046 | +1.05% (increase)               |
| H200 (sm_90a)       |      512 | depthwise   |              85.937 |                     85.470 | -0.54%                          |
| H200 (sm_90a)       |      512 | lossguide   |              86.205 |                     87.015 | +0.94% (increase)               |
| RTX 4070tis (sm_89) |      256 | depthwise   |              22.075 |                     21.298 | -3.52%                          |
| RTX 4070tis (sm_89) |      256 | lossguide   |              22.173 |                     21.882 | -1.32%                          |
| RTX 4070tis (sm_89) |      512 | depthwise   |              46.496 |                     43.842 | -5.71%                          |
| RTX 4070tis (sm_89) |      512 | lossguide   |              46.524 |                     45.596 | -2.00%                          |
| DGX Spark (sm_121)  |      256 | depthwise   |              47.530 |                     45.715 | -3.82%                          |
| DGX Spark (sm_121)  |      256 | lossguide   |              47.813 |                     46.926 | -1.85%                          |
| DGX Spark (sm_121)  |      512 | depthwise   |             105.833 |                     96.269 | -9.04%                          |
| DGX Spark (sm_121)  |      512 | lossguide   |             106.111 |                    105.797 | -0.30%                          |

All 14 cases with Train-rmse recorded for both branches match exactly at the logged precision (five decimal places). No accuracy discrepancy is present in those paired values. This checks final training RMSE; the log does not establish exact model equality or held-out accuracy.

The recorded training-time increases are H200 (sm_90a), 256 features, `lossguide` (+1.05%); H200 (sm_90a), 512 features, `lossguide` (+0.94%). The other 14 cases show decreases. Each branch has one recorded timing per case, so the table does not measure run-to-run variation.

## NVIDIA RTX PRO 6000 Blackwell Server Edition

### 76c515930, 4cc8d8ec0, and 05d41a90c vs. master

Sources: [master results](</home/jiamingy/ws/xgboost_dev/bench/Blackwell Pro 6000/blwp6000-master.zip>), [76c515930 results](</home/jiamingy/ws/xgboost_dev/bench/Blackwell Pro 6000/blwp6000-76c515930.zip>), [4cc8d8ec0 results](</home/jiamingy/ws/xgboost_dev/bench/Blackwell Pro 6000/rtx6000-4cc8d8ec0.zip>), and [05d41a90c results](</home/jiamingy/ws/xgboost_dev/bench/Blackwell Pro 6000/rtx6000-05d41a90c.zip>). The archive names identify the revisions; the JSON files do not record XGBoost commit hashes. After the master baseline, columns follow commit order: `76c515930` (2026-09-29 06:49 +08:00), `4cc8d8ec0` (2026-10-02 16:09 +08:00), then `05d41a90c` (2026-10-02 18:04 +08:00).

These runs use `qdm-iter` with 64 batches of 2^20 samples (67,108,864 samples), 128 boosting rounds, maximum depth 6, and 256 bins. Single-target runs use `one_output_per_tree`; vector-leaf runs use `multi_output_tree` with 4 targets and `debug_synchronize=true`. Recorded parameters, machine metadata, versions, and build information match for every paired case. Times use `timer["Train"]["Train"]`, excluding DMatrix construction and data generation; changes are `100 × (revision / master − 1)`, calculated from full precision.

| Features | Grow policy | Target configuration    | Master training (s) | 76c515930 training (s) | 76c515930 change vs. master | 4cc8d8ec0 training (s) | 4cc8d8ec0 change vs. master | 05d41a90c training (s) | 05d41a90c change vs. master |
|---------:|-------------|-------------------------|--------------------:|-----------------------:|-----------------------------|-----------------------:|-----------------------------|-----------------------:|-----------------------------|
|      256 | depthwise   | Single target           |              34.816 |                 32.255 | -7.36%                      |                 32.421 | -6.88%                      |                 32.222 | -7.45%                      |
|      256 | depthwise   | Vector leaf (4 targets) |             112.670 |                211.744 | +87.93% (increase)          |                125.350 | +11.25% (increase)          |                124.181 | +10.22% (increase)          |
|      256 | lossguide   | Single target           |              36.718 |                 35.376 | -3.65%                      |                 38.613 | +5.16% (increase)           |                 38.479 | +4.80% (increase)           |
|      256 | lossguide   | Vector leaf (4 targets) |             117.478 |                411.473 | +250.26% (increase)         |                132.984 | +13.20% (increase)          |                132.033 | +12.39% (increase)          |
|      512 | depthwise   | Single target           |              88.439 |                 70.005 | -20.84%                     |                 81.451 | -7.90%                      |                 81.029 | -8.38%                      |
|      512 | depthwise   | Vector leaf (4 targets) |             273.329 |                465.816 | +70.42% (increase)          |                288.468 | +5.54% (increase)           |                285.610 | +4.49% (increase)           |
|      512 | lossguide   | Single target           |              78.502 |                 76.034 | -3.14%                      |                 83.343 | +6.17% (increase)           |                 82.874 | +5.57% (increase)           |
|      512 | lossguide   | Vector leaf (4 targets) |             255.208 |                438.645 | +71.88% (increase)          |                289.616 | +13.48% (increase)          |                287.081 | +12.49% (increase)          |

For `76c515930`, single-target training time decreases in all four cases (3.14–20.84%). Vector-leaf training time increases in all four cases (70.42–250.26%).

For `4cc8d8ec0`, single-target training time decreases by 6.88–7.90% with `depthwise` and increases by 5.16–6.17% with `lossguide`. Vector-leaf training time increases in all four cases (5.54–13.48%).

For `05d41a90c`, single-target training time decreases by 7.45–8.38% with `depthwise` and increases by 4.80–5.57% with `lossguide`. Vector-leaf training time increases in all four cases (4.49–12.49%).

Training RMSE matches exactly at the precision stored in the JSON files at every boosting round across all four revisions for all eight cases. Each revision has one recorded timing per case, so these results do not measure run-to-run variation.

## NVIDIA A100 80GB PCIe (sm_80)

### 53d2a5445 vs. master

Sources: [master results](/home/jiamingy/ws/xgboost_dev/bench/A100/a100-master.zip) and [53d2a5445 results](/home/jiamingy/ws/xgboost_dev/bench/A100/a100-53d2a5445.zip). The archives are labeled `master` and `53d2a5445`; neither records an XGBoost commit hash in its JSON files.

These runs use `qdm-iter` with 64 batches of 2^20 samples (67,108,864 samples), 128 boosting rounds, maximum depth 6, and 256 bins. Single-target runs use `one_output_per_tree`; vector-leaf runs use `multi_output_tree` with 4 targets and `debug_synchronize=true`. Recorded parameters, machine metadata, versions, and build information match for every paired case. Times use `timer["Train"]["Train"]`, excluding DMatrix construction and data generation; changes are `100 × (53d2a5445 / master − 1)`, calculated from full precision.

| Features | Grow policy | Target configuration    | Master training (s) | 53d2a5445 training (s) | Training-time change vs. master |
|---------:|-------------|-------------------------|--------------------:|-----------------------:|---------------------------------|
|      256 | depthwise   | Single target           |              78.305 |                 64.347 | -17.82%                         |
|      256 | depthwise   | Vector leaf (4 targets) |             233.895 |                238.660 | +2.04% (increase)               |
|      256 | lossguide   | Single target           |              78.796 |                 65.111 | -17.37%                         |
|      256 | lossguide   | Vector leaf (4 targets) |             235.448 |                235.236 | -0.09%                          |
|      512 | depthwise   | Single target           |             168.911 |                141.580 | -16.18%                         |
|      512 | depthwise   | Vector leaf (4 targets) |             514.659 |                545.381 | +5.97% (increase)               |
|      512 | lossguide   | Single target           |             169.169 |                140.394 | -17.01%                         |
|      512 | lossguide   | Vector leaf (4 targets) |             515.760 |                530.969 | +2.95% (increase)               |

Single-target training time decreases in all four cases (16.18–17.82%). Vector-leaf training time decreases by 0.09% for 256 features with `lossguide`; the other three vector-leaf cases increase by 2.04–5.97%. Training RMSE matches exactly at the precision stored in the JSON files at every boosting round for all eight pairs. Each branch has one recorded timing per case, so these results do not measure run-to-run variation.

## DGX Spark (NVIDIA GB10)

### 53d2a5445 and 05d41a90c vs. master

Sources: [master results](</home/jiamingy/ws/xgboost_dev/bench/DGX Spark/spark-master.zip>), [53d2a5445 results](</home/jiamingy/ws/xgboost_dev/bench/DGX Spark/spark-53d2a5445.zip>), and [05d41a90c results](</home/jiamingy/ws/xgboost_dev/bench/DGX Spark/spark-05d41a90c.zip>). The archive names identify the revisions; the JSON files do not record XGBoost commit hashes. After the master baseline, columns follow commit order: `53d2a5445` (2026-09-30), then `05d41a90c` (2026-10-02).

These runs use `qdm-iter` with 16 batches of 2^20 samples (16,777,216 samples), 128 boosting rounds, maximum depth 6, and 256 bins. Single-target runs use `one_output_per_tree`; vector-leaf runs use `multi_output_tree` with 4 targets and `debug_synchronize=true`. Recorded parameters, machine metadata, versions, and build information match for every paired case. Times use `timer["Train"]["Train"]`, excluding DMatrix construction and data generation; changes are `100 × (revision / master − 1)`, calculated from full precision.

| Features | Grow policy | Target configuration    | Master training (s) | 53d2a5445 training (s) | 53d2a5445 change vs. master | 05d41a90c training (s) | 05d41a90c change vs. master |
|---------:|-------------|-------------------------|--------------------:|-----------------------:|-----------------------------|-----------------------:|-----------------------------|
|      256 | depthwise   | Single target           |              47.312 |                 45.795 | -3.21%                      |                 45.902 | -2.98%                      |
|      256 | depthwise   | Vector leaf (4 targets) |             153.611 |                160.625 | +4.57% (increase)           |                160.622 | +4.56% (increase)           |
|      256 | lossguide   | Single target           |              48.060 |                 47.749 | -0.65%                      |                 47.537 | -1.09%                      |
|      256 | lossguide   | Vector leaf (4 targets) |             155.617 |                158.754 | +2.02% (increase)           |                157.905 | +1.47% (increase)           |
|      512 | depthwise   | Single target           |             105.081 |                 96.579 | -8.09%                      |                 97.547 | -7.17%                      |
|      512 | depthwise   | Vector leaf (4 targets) |             392.654 |                373.626 | -4.85%                      |                373.486 | -4.88%                      |
|      512 | lossguide   | Single target           |             105.779 |                105.784 | +0.00% (increase)           |                105.809 | +0.03% (increase)           |
|      512 | lossguide   | Vector leaf (4 targets) |             533.442 |                403.416 | -24.37%                     |                404.122 | -24.24%                     |

For `53d2a5445`, single-target training time decreases in three cases (0.65–8.09%); the 512-feature `lossguide` case increases by 0.0042%, which rounds to +0.00% in the table. Vector-leaf training time decreases by 4.85–24.37% with 512 features and increases by 2.02–4.57% with 256 features.

For `05d41a90c`, single-target training time decreases in three cases (1.09–7.17%); the 512-feature `lossguide` case increases by 0.03%. Vector-leaf training time decreases by 4.88–24.24% with 512 features and increases by 1.47–4.56% with 256 features.

Training RMSE matches exactly at the precision stored in the JSON files at every boosting round across all three revisions for all eight cases. Each revision has one recorded timing per case, so these results do not measure run-to-run variation.

## In-core vs. external-memory training (depthwise)

Source: [ivse.log](/home/jiamingy/ws/xgboost_dev/bench/ivse.log). All nine runs use `depthwise` and 128 boosting rounds. Times are from `Train Train ended in`, excluding DMatrix construction, and are rounded to three decimal places.

Each change is relative to `qdm-iter` at the same feature count: `100 × (ext-qdm-iter / qdm-iter − 1)`, calculated from the full recorded precision. Positive values indicate increased training time.

| Features | qdm-iter (s) | ext-qdm-iter, cache_host_ratio=0.0 (s) | Change vs. qdm-iter | ext-qdm-iter, cache_host_ratio=1.0 (s) | Change vs. qdm-iter | ext-qdm, pre-alloc, CHR=1.0 |         |
|---------:|-------------:|---------------------------------------:|--------------------:|---------------------------------------:|--------------------:|-----------------------------|---------|
|      256 |       40.002 |                                 41.394 |              +3.48% |                                 52.008 |             +30.01% | 51.362                      |         |
|      512 |       85.466 |                                 88.725 |              +3.81% |                                103.861 |             +21.52% | 98.18                       | +10.65% |
|     1024 |      177.817 |                                184.323 |              +3.66% |                                193.996 |              +9.10% | 194.346                     |         |

Final training RMSE matches across all three variants at each feature count, at the logged precision of five decimal places.

## In-core vs. external-memory training (depthwise, master)

Source: [ivse-master.log](/home/jiamingy/ws/xgboost_dev/bench/ivse-master.log), run on the `master` branch. All nine runs use `depthwise` and 128 boosting rounds. Times are from `Train Train ended in`, excluding DMatrix construction, and are rounded to three decimal places.

Each change is relative to `qdm-iter` at the same feature count: `100 × (ext-qdm-iter / qdm-iter − 1)`, calculated from the full recorded precision. Positive values indicate increased training time.

| Features | qdm-iter (s) | ext-qdm-iter, cache_host_ratio=0.0 (s) | Change vs. qdm-iter | ext-qdm-iter, cache_host_ratio=1.0 (s) | Change vs. qdm-iter |
|---------:|-------------:|---------------------------------------:|--------------------:|---------------------------------------:|--------------------:|
|      256 |       40.415 |                                 42.203 |              +4.43% |                                 54.868 |             +35.76% |
|      512 |       85.956 |                                 93.097 |              +8.31% |                                106.024 |             +23.35% |
|     1024 |      179.365 |                                205.325 |             +14.47% |                                231.360 |             +28.99% |

Final training RMSE matches across all three variants at each feature count and matches the corresponding cases in `ivse.log`, at the logged precision of five decimal places.

## Missing values and sparse data (f58ad8000 vs. master)

Measured on 2026-09-30. The branch is `f58ad8000` ("Fix all missing."), and master is `c2ca8c99a`. Both are `RelWithDebInfo` builds for sm_89 with `USE_NVTX=ON`. `USE_NVTX` adds `-lineinfo`, which doubles the embedded device code and adds about 35 ms of module loading to the first iteration. Both builds must use the same setting, otherwise short runs are skewed.

All runs use `dxgb-bench bench --task=qdm-iter --fly --device=cuda --mr=cuda --tree_method=hist --policy=depthwise --max_depth=6 --n_bins=256 --n_rounds=16`, with 8 batches of 2^20 samples (8,388,608 samples). `--sparsity` sets the fraction of missing values. The page is dense compressed only if at least one row has no missing values. Single-target runs use `one_output_per_tree`. `dxgb-bench` enables `debug_synchronize` for `multi_output_tree`, and that check fails on both branches when trees have +inf split conditions. 4-target runs use `multi_output_tree`.

Times are `Train Train ended in`, taking the median of three runs that alternate the two libraries. Changes are `100 × (f58ad8000 / master − 1)`, calculated from full precision. With 256 features, the sparse page is a single group whose histogram doesn't fit in shared memory, so it uses global memory. With 16 features, the sparse histogram fits in shared memory.

| GPU (architecture)        | Features | Sparsity | Ellpack layout   | Targets | Master training (s) | f58ad8000 training (s) | Training-time change vs. master |
|---------------------------|---------:|---------:|------------------|--------:|--------------------:|-----------------------:|---------------------------------|
| RTX 4070 Ti SUPER (sm_89) |      256 |     0.01 | Dense compressed |       1 |               2.056 |                  1.924 | -6.41%                          |
| RTX 4070 Ti SUPER (sm_89) |      256 |     0.03 | Dense compressed |       1 |               1.988 |                  1.860 | -6.45%                          |
| RTX 4070 Ti SUPER (sm_89) |      256 |      0.1 | Sparse           |       1 |               4.679 |                  4.686 | +0.16% (increase)               |
| RTX 4070 Ti SUPER (sm_89) |      256 |      0.5 | Sparse           |       1 |               2.396 |                  2.409 | +0.52% (increase)               |
| RTX 4070 Ti SUPER (sm_89) |      256 |      0.9 | Sparse           |       1 |               0.595 |                  0.598 | +0.40% (increase)               |
| RTX 4070 Ti SUPER (sm_89) |       16 |     0.05 | Dense compressed |       1 |               0.293 |                  0.307 | +4.85% (increase)               |
| RTX 4070 Ti SUPER (sm_89) |       16 |      0.7 | Sparse           |       1 |               0.296 |                  0.295 | -0.42%                          |
| RTX 4070 Ti SUPER (sm_89) |      256 |     0.01 | Dense compressed |       4 |               6.159 |                  5.914 | -3.98%                          |
| RTX 4070 Ti SUPER (sm_89) |      256 |     0.03 | Dense compressed |       4 |               5.944 |                  5.818 | -2.12%                          |
| RTX 4070 Ti SUPER (sm_89) |      256 |      0.1 | Sparse           |       4 |              17.212 |                 16.891 | -1.87%                          |
| RTX 4070 Ti SUPER (sm_89) |      256 |      0.5 | Sparse           |       4 |               8.048 |                  7.897 | -1.88%                          |
| RTX 4070 Ti SUPER (sm_89) |      256 |      0.9 | Sparse           |       4 |               1.662 |                  1.685 | +1.41% (increase)               |
| RTX 4070 Ti SUPER (sm_89) |       16 |     0.05 | Dense compressed |       4 |               0.674 |                  0.680 | +0.90% (increase)               |
| RTX 4070 Ti SUPER (sm_89) |       16 |      0.7 | Sparse           |       4 |               0.698 |                  0.729 | +4.56% (increase)               |

Histogram kernel time is the total GPU time of the histogram kernels in an `nsys profile --trace=cuda` run: `StHistKernel` and `MtHistKernel` on master, `HistogramKernel` on the branch. Each case was profiled once, and the settings match the table above. These profiles used the master build without `USE_NVTX`, which doesn't change the device code.

| GPU (architecture)        | Features | Sparsity | Targets | Master histogram (ms) | f58ad8000 histogram (ms) | Histogram-time change vs. master |
|---------------------------|---------:|---------:|--------:|----------------------:|-------------------------:|----------------------------------|
| RTX 4070 Ti SUPER (sm_89) |      256 |     0.01 |       1 |                1725.9 |                   1590.0 | -7.87%                           |
| RTX 4070 Ti SUPER (sm_89) |      256 |      0.1 |       1 |                3925.4 |                   3936.0 | +0.27% (increase)                |
| RTX 4070 Ti SUPER (sm_89) |      256 |      0.9 |       1 |                 250.6 |                    250.3 | -0.12%                           |
| RTX 4070 Ti SUPER (sm_89) |       16 |     0.05 |       1 |                  86.6 |                     86.8 | +0.23% (increase)                |
| RTX 4070 Ti SUPER (sm_89) |       16 |      0.7 |       1 |                  68.3 |                     68.0 | -0.44%                           |
| RTX 4070 Ti SUPER (sm_89) |      256 |      0.9 |       4 |                 985.4 |                    981.6 | -0.39%                           |
| RTX 4070 Ti SUPER (sm_89) |       16 |     0.05 |       4 |                 238.2 |                    245.0 | +2.85% (increase)                |
| RTX 4070 Ti SUPER (sm_89) |       16 |      0.7 |       4 |                 251.2 |                    258.9 | +3.07% (increase)                |

Dense compressed data with missing values is 2–6% faster to train with 256 features. Sparse data is 0.16–0.52% slower than master for a single target. With 4 targets it is up to 1.9% faster, except at sparsity 0.9 (+1.41%). The 16-feature runs take 0.3–0.7 s, and their run-to-run spread is about ±3%. The single-target +4.85% has no matching change in histogram time. With 4 targets on 16 features, histogram time is about 3% higher.

## Imbalanced feature groups (f58ad8000 vs. master)

Measured on 2026-09-30 with the same builds and timing method as the preceding section. The data has 3,072 binary features followed by 1,008 continuous features (standard normal), in 4 batches of 131,072 samples (524,288 samples). Binary features get 3 bins and continuous features 257 bins. On sm_89, the single-target shared memory budget (101,376 bytes, 6,336 bins) gives 44 groups: 2,112 and 973 features in the first two, then 42 groups of 24 features. The multi-target budget (50,176 bytes, 3,136 bins) gives 87 groups: three of about 1,045 features and 84 of 12.

All runs use `dxgb-bench bench --task=qdm-iter --loadfrom=<data> --device=cuda --mr=cuda --tree_method=hist --n_bins=256 --n_rounds=16` with the grow policy and `--max_depth` shown. The data scripts are `gen.py` and `gen_y.py` in `/tmp/imb`, which is not persistent. Single-target runs use `one_output_per_tree` and 4-target runs use `multi_output_tree`.

| GPU (architecture)        | Grow policy | Max depth | Targets | Master training (s) | f58ad8000 training (s) | Training-time change vs. master |
|---------------------------|-------------|----------:|--------:|--------------------:|-----------------------:|---------------------------------|
| RTX 4070 Ti SUPER (sm_89) | depthwise   |         6 |       1 |               1.230 |                  1.113 | -9.53%                          |
| RTX 4070 Ti SUPER (sm_89) | depthwise   |         8 |       1 |               2.259 |                  1.785 | -20.97%                         |
| RTX 4070 Ti SUPER (sm_89) | lossguide   |         6 |       1 |               1.260 |                  1.167 | -7.37%                          |
| RTX 4070 Ti SUPER (sm_89) | lossguide   |         8 |       1 |               2.372 |                  2.030 | -14.42%                         |
| RTX 4070 Ti SUPER (sm_89) | depthwise   |         6 |       4 |               4.974 |                  3.733 | -24.94%                         |
| RTX 4070 Ti SUPER (sm_89) | depthwise   |         8 |       4 |               8.009 |                  6.082 | -24.06%                         |
| RTX 4070 Ti SUPER (sm_89) | lossguide   |         6 |       4 |               4.936 |                  3.872 | -21.55%                         |
| RTX 4070 Ti SUPER (sm_89) | lossguide   |         8 |       4 |               8.294 |                  6.499 | -21.64%                         |

Histogram kernel time, measured the same way as in the preceding section, one profile per case:

| GPU (architecture)        | Grow policy | Max depth | Targets | Master histogram (ms) | f58ad8000 histogram (ms) | Histogram-time change vs. master |
|---------------------------|-------------|----------:|--------:|----------------------:|-------------------------:|----------------------------------|
| RTX 4070 Ti SUPER (sm_89) | depthwise   |         8 |       1 |                1479.4 |                   1024.6 | -30.74%                          |
| RTX 4070 Ti SUPER (sm_89) | lossguide   |         6 |       1 |                 966.8 |                    858.0 | -11.25%                          |
| RTX 4070 Ti SUPER (sm_89) | lossguide   |         8 |       1 |                1473.4 |                   1115.4 | -24.30%                          |
| RTX 4070 Ti SUPER (sm_89) | depthwise   |         6 |       4 |                4023.3 |                   2863.7 | -28.82%                          |
| RTX 4070 Ti SUPER (sm_89) | lossguide   |         8 |       4 |                5084.6 |                   3282.0 | -35.45%                          |

Training time decreases in all eight cases (7.37–24.94%). The histogram kernel accounts for the decrease. Lossguide at depth 8 builds one or two nodes per launch, down to a few thousand samples per node, and still gains 24–35% in histogram time.
