# GPU histogram launch policy: implementation plan

Status: proposal. Target file: `src/tree/gpu_hist/histogram.cu`.

## 1. Problem

On an RTX PRO 6000 Blackwell Server Edition (188 SMs, sm_120), 67,108,864 rows,
`max_depth=6`, `max_bin=256`, 128 rounds, `multi_strategy="multi_output_tree"` with 4
targets, the `optimize-hist` branch is slower than master at `grow_policy="depthwise"`:

| features | targets | grow policy | master (s) | branch (s) | change |
|---:|---:|---|---:|---:|---:|
| 256 | 4 | depthwise | 102.07 | 113.35 | **+11.1 %** |
| 512 | 4 | depthwise | 212.25 | 257.52 | **+21.3 %** |
| 256 | 4 | lossguide | 100.01 | 101.14 | +1.1 % |
| 512 | 4 | lossguide | 214.72 | 212.28 | −1.1 % |
| 256/512 | 1 | both | — | — | −4.6 % … −7.3 % |

The per-round RMSE sequences are bitwise identical, so the trees are the same and no extra
work is being done. The difference is entirely in how the histogram kernel's work is
partitioned across blocks.

Master's histogram policy (`f9bdf8f7b`) is byte-identical to `c2ca8c99a`, so "master" and
"old" are interchangeable below.

## 2. Definitions

| name | meaning | value in the 512-feature case |
|---|---|---|
| `n_features` | columns (= `row_stride`, dense) | 512 |
| `n_targets` | targets per tree (vector leaf) | 4 |
| `max_bin` | bins per feature | 256 |
| `bytes_per_entry` | gidx symbol width / 8 | 1 |
| `features_per_group` | columns whose bins fit one block's privatized histogram | 12 |
| `n_groups` | `ceil(n_features / features_per_group)` | 43 |
| `n_resident_blocks` | `n_blks_per_mp * n_mps` (occupancy × SMs) | 2 × 188 = 376 |
| `kSectorBytes` | DRAM/L2 fetch granularity | 32 |
| `ShmemSize()` | bytes of privatized histogram per block | 49152 |
| `entries_per_chunk` | gidx entries one block processes (**the decision variable**) | — |

## 3. Diagnosis

Each gidx byte is needed more than once, along two independent axes:

| axis | same bytes needed by | satisfied when |
|---|---|---|
| **targets** | the `n_targets` blocks of one chunk | their shared footprint survives in L2 while they drift apart |
| **feature groups** | the groups reading the same rows — a 32 B sector holds `features_per_group` useful bytes, so `kSectorBytes / (features_per_group * bytes_per_entry)` ≈ 2.7 groups must read a row concurrently to use the sector fully | enough groups are **co-resident** on the same rows |

**Master** partitions per `(node, feature group, chunk, target)` with `gridDim.y = n_groups`.
Because `gridDim.x` (48,128 at the deepest level) far exceeds the 376 resident slots, every
co-resident block shares `blockIdx.y` — exactly one feature group is in flight at a time.
Master therefore has the target axis (blocks are ~5 tiles long, so siblings cannot drift) and
structurally cannot have the group axis. Its blocks are so short that it emits ~2.07 M of them
per level and ~95 GB of L2-resident flush atomics.

**The branch** concatenates all `(node, group)` segments into one item space, nested
`node -> group -> row -> feature`, and cuts it into equal chunks:

```cpp
resident      = n_blks_per_mp * n_mps / n_targets;
tiles_per_blk = max(DivRoundUp(n_tiles, resident * kMaxWaves), min(32, ...));  // kMaxWaves = 32
```

Two consequences:

1. The resident reused footprint is exactly `launch_bytes / kMaxWaves`, **independent of L2,
   SM count, `n_targets`, `n_features` and bin width.** The policy contains no cache term. It
   exceeds L2 whenever a launch touches more than `kMaxWaves * L2` ≈ 2.4 GB on this part; a
   depthwise level touches 17.2 GB, i.e. 7.7× over.
2. Group co-residency is `min(n_groups, resident * chunk_entries / segment_entries)`, which is
   **1.3 at level 1 and 42 at level 6**. The branch only gets the group axis where its chunks
   happen to exceed `segment / resident` — deep in the tree, and only because its chunks are
   huge. Those huge chunks are what break the target axis.

Single target is unaffected because there is no target axis to lose, and because `resident` is
not divided by `n_targets`, so its chunks are 4× shorter. Lossguide is unaffected because it
pops one node per launch, so `launch_bytes` — and therefore chunk length — shrinks with depth.

### Supporting measurements

Paired launch-by-launch inside one process (alternate the configuration on a static counter,
group the nsys trace by grid size), so clock throttling cancels:

- Moving the target index to the outer grid dimension at fixed block length costs **23 %** →
  the cross-target sharing is real and load-bearing.
- The branch/master ratio is monotone in block length: 1.16 at 1902 tiles, 1.11 at 476,
  **1.03 at 119**, 1.04 at 59, and 1.86 at 1 tile (flush-bound).
- Local reproduction on a 46 SM sm_120 part: +4…+11 % for 4-target depthwise, +0 % for single
  target.

### Nesting order, fastest-varying to slowest

| | fastest | | | slowest |
|---|---|---|---|---|
| master | target | chunk (grid-strided tiles inside one segment) | node | **group** |
| branch | target | **row position inside a group** | **group** | node |
| proposal | target | **group** | **chunk** | node |

## 4. Design

Scope: launch policy only. `targets_per_block = 1`, chunking in **entries**, no rows anywhere.
`FeatureGroups`, the shared-memory budget, launch bounds and `HistKernelSegment` are untouched.

**Work unit.** `(node, feature group, entry-chunk, target)`. A chunk is `entries_per_chunk`
contiguous entries *inside* one `(node, group)` segment, so no block spans two segments.

**Ordering.** target fastest, then group, then chunk, then node. Rectangular grid, decoded by
division, no prefix-sum array:

```cpp
target_idx = blockIdx.x % n_targets;
group_idx  = (blockIdx.x / n_targets) % n_groups;
chunk_idx  = (blockIdx.x / (n_targets * n_groups)) % n_chunks_per_segment;
node_idx   =  blockIdx.x / (n_targets * n_groups * n_chunks_per_segment);
```

### Invariants this buys

All independent of `entries_per_chunk`:

| quantity | value |
|---|---|
| co-resident groups | `min(n_groups, n_resident_blocks / n_targets)` — was `min(n_groups, n_resident_chunks * chunk / segment_g)`, which varies 1.3–43 by level and per group under width skew |
| work per non-empty block | exactly `entries_per_chunk` — so skewed group widths cause **no** tail (this is why entries beat rows) |
| flush ratio | `n_targets * ShmemSize() / (entries_per_chunk * bytes_per_entry)` — grouping-independent, since every group is packed to the bin budget |
| device arrays per launch | `ridx_iters`, `hists` — `sizes_csum` is dropped (32 KB instead of 40 KB at `n_nodes = 1024`) |

### Known costs

- Front alignment is exact only within a group-width class. `FeatureGroups` packs by bin count,
  so widths vary: narrow groups cluster into 1–2 widths and wide groups
  (`>= kSectorBytes / bytes_per_entry` features) fill sectors alone, so the classes that need
  alignment have it. Measured on synthetic skew: narrow-group widths were `{12}` and `{12, 18}`.
- Empty blocks from the rectangular grid, bounded by the node-size × width skew. They exit
  before touching shared memory, so the cost is dispatch only.

---

## 5. Tunables

The plan reduces the constant count from three to two, and lands with **no fitted constants**.

### 5.1 Taxonomy

**Derived — no freedom**

- `cap_one_wave = total_entries * n_targets / n_resident_blocks`
- the `n_targets * ShmemSize()` factor in the flush floor
- the grid guard (a hardware limit)

**Declared budgets — free, but exactly verifiable on the host before every launch**

- `kMinWaves` — how much tail is acceptable. Observable as `blocks / n_resident_blocks`.
- `kMaxFlushRatio` — how much flush is acceptable. `flush_bytes` is computable exactly as
  `total_entries * n_targets * ShmemSize() / entries_per_chunk`, so the budget can be asserted,
  not guessed.

**Fitted constants — need measurement, can be wrong invisibly**

- `l2_usable_fraction` — deferred to Phase 4; the Phase 3 sweep may say never.
- `kAtomicToDramRatio` — **not adopted**, see 5.3.

### 5.2 Two existing constants become formulas

`kMinTiles = 32` is not an independent choice. It equals
`n_targets * ShmemSize() / (1.0 * kTileSize * bytes_per_entry)` for the benchmark shape, i.e.
"flush <= 1x the gidx bytes" at 4 targets with a 49 KB group. Generalising it is a fix:

| `n_targets` | correct floor | today's literal |
|---:|---:|---:|
| 1 | 8 tiles | 32 (4× too conservative) |
| 4 | 32 tiles | 32 |
| 32 | 256 tiles | 32 (8× too small) |

`kItemsPerThread = 8` / `kTileSize` lose their meaning after Phase 2 — the inner loop is a plain
block-stride loop, so they only serve as a rounding unit. Round `entries_per_chunk` to
`kBlockThreads` instead and delete both.

Net: today `kMaxWaves`, `kMinTiles`, `kItemsPerThread` → proposal `kMinWaves`,
`kMaxFlushRatio`, both generalised.

### 5.3 Rejected: collapsing to one parameter

The flush ↔ tail trade is one degree of freedom bounded from both sides, so it has a closed form
with a single hardware ratio:

```
entries_per_chunk = sqrt(2 * kAtomicToDramRatio * cap_one_wave * ShmemSize() / bytes_per_entry)
```

sqrt-damped, so a 4× error in the ratio moves the answer 2× (172 / 345 / 690 tiles at ratio
0.0625 / 0.25 / 1.0). But it disagrees with the two clamps most in the external-memory regime,
and in the opposite direction:

| rows/launch | 2-clamp t/blk | flush | waves | sqrt t/blk | flush | waves |
|---:|---:|---:|---:|---:|---:|---:|
| 0.26 M | 128 | **25 %** | 1.8 | 30 | **107 %** | 7.8 |
| 1.05 M | 128 | **25 %** | 7.3 | 61 | **52 %** | 15.2 |
| 4.19 M | 128 | 25 % | 29.1 | 122 | 26 % | 30.5 |
| 33.55 M | 930 | 3 % | 32.0 | 345 | 9 % | 86.2 |

The sqrt form embeds a tail model (`tail ∝ entries_per_chunk`, half a block per SM) for which
there is no measurement. It trades a verifiable budget for an unverifiable model to save one
constant. Revisit only if the Phase 3 external-memory sweep shows the optimum scaling as
`sqrt(launch_size)` — the data would then supply the ratio.

---

## 6. Phase 1 — bound priority in `SliceItems` (standalone)

Independent of the ordering change. Helps external memory on master and the branch alike.
Land and measure first.

Today's

```cpp
min_tiles = std::min(kMinTiles, DivRoundUp(n_tiles, n_resident_blks_per_target));
```

**lowers** the flush floor for small launches in order to fill the device. With external memory
each page is its own launch, so this is exactly the regime where it hurts: at a 1 M-row page the
kernel spends as many bytes on flushes as on reading the data.

```cpp
// src/tree/gpu_hist/histogram.cu, HistKernel::SliceItems
static constexpr double      kMaxFlushRatio = 0.25;
static constexpr std::size_t kMinWaves      = 32;   // was kMaxWaves; it was always a lower bound

// derived floor, replaces the literal kMinTiles
auto floor_flush  = DivRoundUp(n_targets * shmem_bytes_per_block,
                               static_cast<std::size_t>(kMaxFlushRatio * Policy::kTileSize * bpe));
auto cap_balance  = std::max<std::size_t>(1, DivRoundUp(n_tiles, res * kMinWaves));
auto cap_one_wave = std::max<std::size_t>(1, DivRoundUp(n_tiles * n_targets, n_resident_blocks));

auto n_tiles_per_blk = std::min(std::max(cap_balance, floor_flush), cap_one_wave);
```

Thread `shmem_bytes_per_block` (= `feature_groups.ShmemSize()`), `n_targets` and
`bytes_per_entry` into `SliceItems`. Phase 1 stays in tiles to keep the diff minimal; Phase 2
converts to entries and drops `kTileSize`.

Effect, 512 features / 4 targets / 188 SMs:

| rows per launch | today t/blk | blocks | waves | flush | after t/blk | blocks | waves | flush | binding |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.03 M | 30 | 368 | 1.0 | 107 % | 30 | 368 | 1.0 | 107 % | one-wave |
| 0.26 M | 32 | 2,732 | 7.3 | **100 %** | 128 | 684 | 1.8 | **25 %** | flush floor |
| 1.05 M | 32 | 10,924 | 29.1 | **100 %** | 128 | 2,732 | 7.3 | **25 %** | flush floor |
| 4.19 M | 117 | 11,952 | 31.8 | 27 % | 128 | 10,924 | 29.1 | 25 % | flush floor |
| 16.78 M | 465 | 12,028 | 32.0 | 7 % | 465 | 12,028 | 32.0 | 7 % | balance |
| 33.55 M | 930 | 12,028 | 32.0 | 3 % | 930 | 12,028 | 32.0 | 3 % | balance |

In-core is unchanged (`floor_flush = 128` tiles never binds against `cap_balance = 930`). Single
target also improves on small launches, since its derived floor is 8 tiles rather than 32.

**Gate:** external-memory benchmarks (`ext-qdm-iter`, varying `--n_batches`) improve or are
neutral; in-core timings unchanged.

---

## 7. Phase 2 — decomposition and ordering

### 7.1 Host side, `HistKernel::DispatchHist`

```cpp
// from h_feature_groups.feature_segments.HostVector(); FeatureGroups itself is not modified
bst_feature_t max_group_features = max adjacent difference, or matrix.row_stride when !kCompressed;
bst_idx_t     max_node_rows      = max over h_ridx_iters of .size();

// Phase 1 bounds, now in entries, rounded to kBlockThreads
bst_idx_t entries_per_chunk = /* min(max(cap_balance, floor_flush), cap_one_wave) */;
auto n_chunks_per_segment = DivRoundUp(max_node_rows * max_group_features, entries_per_chunk);

// grid guard: raise entries_per_chunk until the grid fits
std::uint64_t n_blks = n_nodes * n_groups * n_chunks_per_segment * n_targets;
while (n_blks > kMaxGridX) { entries_per_chunk *= 2; /* recompute */ }
CHECK_LE(n_blks, std::numeric_limits<std::uint32_t>::max());
```

Delete the `sizes_csum` `TemporaryArray` and its `dh::CopyTo`. Keep `h_sizes_csum` host-side for
the `back() == 0` early-out.

### 7.2 Kernel

```cpp
template <typename Policy, typename Accessor, typename RidxIterSpan>
__global__ __launch_bounds__(...) void HistogramKernel(
    Accessor const matrix, FeatureGroupsAccessor const feature_groups,
    RidxIterSpan const* d_ridx_iters, common::Span<GradientPairInt64> const* node_hists,
    GradientPairInt64 const* d_gpair, bst_idx_t n_samples, bst_target_t n_targets,
    bst_idx_t entries_per_chunk, std::uint32_t n_chunks_per_segment) {
  if constexpr (Policy::kSingleTarget) { n_targets = 1; }
  auto const n_groups = feature_groups.NumGroups();

  auto target_idx = blockIdx.x % n_targets;
  auto group_idx  = (blockIdx.x / n_targets) % n_groups;
  auto chunk_idx  = (blockIdx.x / (n_targets * n_groups)) % n_chunks_per_segment;
  auto node_idx   =  blockIdx.x / (n_targets * n_groups * n_chunks_per_segment);

  auto group  = feature_groups[group_idx];
  auto d_ridx = d_ridx_iters[node_idx];
  bst_feature_t feature_stride = Policy::kCompressed ? group.num_features : matrix.row_stride;
  bst_idx_t n_entries = d_ridx.size() * feature_stride;
  bst_idx_t begin = chunk_idx * entries_per_chunk;
  if (begin >= n_entries) { return; }                      // empty slot
  bst_idx_t end = min(begin + entries_per_chunk, n_entries);

  extern __align__(std::alignment_of_v<GradientPairInt64>) __shared__ char shmem[];
  auto smem_hist   = reinterpret_cast<GradientPairInt64*>(shmem);
  auto d_node_hist = node_hists[node_idx];
  auto gmem_hist   = d_node_hist.data() + target_idx * (d_node_hist.size() / n_targets);
  __builtin_assume(__isGlobal(gmem_hist));

  if constexpr (Policy::kSharedMem) {
    dh::BlockFill(smem_hist, group.num_bins, GradientPairInt64{});
    __syncthreads();
  }
  HistKernelSegment<Policy>(matrix, group, d_ridx, d_gpair + n_samples * target_idx,
                            smem_hist, gmem_hist, begin, end);
  if constexpr (Policy::kSharedMem) {
    __syncthreads();
    for (auto bin : dh::BlockStrideRange(0, group.num_bins)) {
      AtomicAddGpairGlobal(gmem_hist + group.start_bin + bin, smem_hist[bin]);
    }
  }
}
```

`HistKernelSegment` is called with the **full** row-index span and segment-local `begin`/`end`;
it already derives `ridx_in_set = idx / feature_stride`, so it needs no change and no
`subspan`.

### 7.3 Deleted

`HistSegment`, `FindSegment`, `UpperBoundIdx` (only user), the `while (pos < last())` loop, the
post-barrier `find_segment` recompute, `volatile bst_idx_t pos` and its 8-byte stack frame, the
`sizes_csum` kernel parameter and copy, and `kItemsPerThread` / `HistPolicy::kTileSize` (the
chunk is rounded to `kBlockThreads` instead).

### 7.4 Unchanged

`FeatureGroups` / `feature_groups.cu`, `HistShmemBytes`, `Dft{St,Mt}HistShmemBytes`,
`HistTuning` / launch bounds, the rest of `HistPolicy`, `BlocksPerMp`, `DispatchCudaSm`,
`HistKernelSegment`, `IterIdx`, `LoadGpair`, the atomics, `DeviceHistogramStorage`,
`AllReduceHist`, `SubtractionTrick`, both `BuildHistogram` overloads in `histogram.cuh`, and
both call sites in `updater_gpu_hist.cu{,h}`.

---

## 8. Phase 3 — measurement gate

All variants are bitwise identical (integer atomics), so sweeps carry no correctness risk.

1. **Correctness:** `testxgboost --gtest_filter=*Histogram*` plus `tests/python-gpu`.
2. **Registers:** `./regcheck.sh {75,80,86,90,100,120}` with `parse_regs.py`. Expect ≤ 40
   registers, no spills, and the 8-byte stack frame gone.
3. **In-core:** the four regressing configurations on the RTX PRO 6000, versus master and
   versus the branch.
4. **Per level:** paired in-process A/B over `entries_per_chunk`. The ordering claim predicts
   the optimum **stops moving with tree depth**; under the current ordering the cost model puts
   it 128× apart between levels 1 and 6.
5. **Mechanism:** the `target`-outer switch should still cost ~23 %. If less, the targets were
   not sharing.
6. **External memory:** `--n_batches` sweep. Two questions: empty-block dispatch cost on small
   pages, and whether the optimum follows the clamps or the rejected `sqrt` form of 5.3.

---

## 9. Phase 4 — conditional, only if Phase 3 requires it

- **Locality cap.** If the sweep shows the optimum well below `cap_balance`:

  ```cpp
  l2_budget_bytes = l2_usable_fraction * (l2CacheSize - n_resident_blocks * ShmemSize());
  cap_locality    = l2_budget_bytes * n_targets / (n_resident_blocks * bytes_per_entry);
  ```

  inserted as another `min`. This is the plan's only fitted constant; validate that it tracks
  L2/SM on a second device (H200, 0.38 MB/SM vs 0.68 MB/SM — the predicted optimum differs 2×).

- **Exact chunk map**, if empty-block waste measures significant. Replace the rectangular grid
  with an exclusive scan over the `n_nodes * n_groups` segment chunk counts, computed **on the
  device** from `d_ridx_iters` and `feature_segments` (one small `thrust::exclusive_scan` on the
  stream) and binary-searched in the kernel. Device-to-device, so it does not reintroduce the
  host-to-device copy problem that motivated avoiding a `chunk_ptr` array.

---

## 10. Tests

`tests/cpp/tree/gpu_hist/test_histogram.cu`. The existing cases must pass unmodified — that is
the main safety net, since `HistogramBuildTest` already covers
dense / dense-missing / sparse × 1,3 targets × root/nodes × shared/global × `small_groups`.

Add:

1. **Skewed group widths.** A `HistInput` variant where half the features get `n_bins` and half
   get 4 bins, so `FeatureGroups` produces widths spanning >10×. Asserts correctness and mixed
   widths in `feature_segments`. No current test or benchmark exercises this.
2. **Empty-slot coverage.** Deliberately uneven node sizes (e.g. `{1, 1024, 7, 65536}`) so many
   rectangular slots are empty, including a node with fewer rows than `n_chunks_per_segment`.
3. **Chunk-boundary coverage.** Parametrize `entries_per_chunk` over
   `{kBlockThreads, 2 * kBlockThreads, segment_size - 1, segment_size, segment_size + 1}`
   through a test-only override, to hit the `begin >= n_entries` and partial-last-chunk paths.
4. **Grid guard.** `n_nodes = 1024`, many groups, 32 targets — assert the grid stays under
   `UINT32_MAX` and that `entries_per_chunk` was raised.
5. **Derived floor.** Assert `floor_flush` scales with `n_targets` (8 / 32 / 256 tiles at
   1 / 4 / 32 targets for a 49 KB group), which the old literal `kMinTiles` did not.

## 11. Benchmarks

Add a skewed-bin dataset to the `dxgb_bench` set; all eight current configurations are uniform
dense, which is exactly the case where group-width skew is invisible. Keep the existing eight as
the regression baseline.

## 12. Risks and rollback

| risk | mitigation |
|---|---|
| empty-block dispatch cost on small external-memory pages | measured in 8.6; Phase 4 exact map if it bites |
| grid overflow with many nodes × groups × targets | host guard raises `entries_per_chunk`; test 4 |
| co-residency assumes roughly linear `blockIdx` dispatch, now for `n_groups * n_targets` blocks rather than `n_targets` | 8.5 measures it directly |
| alignment lost across group-width classes | bounded by the width-class analysis; test 1 makes it visible |
| `kMaxFlushRatio = 0.25` is a declared budget, not a measured optimum | the resulting flush volume is computable exactly on the host; 8.6 sweeps it where it matters |
| the cost model behind §4 is unvalidated except for 8.5 | the ordering change follows from the indexing, not from a fitted constant; that is why it lands before any L2 term |

Phases 1 and 2 are separate commits with independent gates. Phase 1 is a ~10-line change to one
function and is reversible on its own. Phase 2 is confined to `HistogramKernel` and
`DispatchHist` and leaves every public signature and every other file unchanged.

## 13. Out of scope

- `targets_per_block > 1` (one block owning several targets' histograms, removing the target
  axis entirely at the cost of narrowing `features_per_group`).
- Changes to `FeatureGroups`, including a `max_features_per_group` cap.
- Making `use_shared` a profitability decision rather than a "does it fit" test.
- The `n_targets` ≳ 24 regime, where the privatized histogram stops paying for itself.
