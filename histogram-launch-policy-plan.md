# GPU histogram launch policy: implementation plan

Status: **Phases 1 and 2 implemented** (§6, §7). Phases 3–4 pending.
Target file: `src/tree/gpu_hist/histogram.cu`.

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

- `kTargetWaves = 32` — the wave count to aim for. Observable as `blocks / n_resident_blocks`.
- `kMinWaves = 4` — the wave count never to go below. Bounds the cost of letting the flush
  budget lengthen blocks: the blocks of a launch are of equal length, so the waste is the
  partially filled last wave, about `1 / (2 * kMinWaves)`.
- `kMaxFlushPercent = 25` — how much flush is acceptable, as a percentage of the gradient
  index bytes a block reads. `flush_bytes` is computable exactly as
  `n_items * n_targets * ShmemSize() / entries_per_blk`, so the budget can be asserted, not
  guessed.

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

Net: today `kMaxWaves`, `kMinTiles`, `kItemsPerThread` → `kTargetWaves`, `kMinWaves`,
`kMaxFlushPercent`, all generalised, with `kItemsPerThread` to be removed in Phase 2. `kMinWaves`
is new and exists only to bound the downside of the flush floor; it replaces an implicit
one-wave cap.

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

## 6. Phase 1 — bound priority in `SliceItems` — IMPLEMENTED

Independent of the ordering change. Helps external memory on master and the branch alike.

Before, the floor was *lowered* for small launches in order to fill the device:

```cpp
min_tiles = std::min(kMinTiles, DivRoundUp(n_tiles, n_resident_blks_per_target));
n_tiles_per_blk = std::max(DivRoundUp(n_tiles, res * kMaxWaves), min_tiles);
```

With external memory each page is its own launch, so this is exactly the regime where it hurts:
at a 1 M-row page the kernel spends as many bytes on flushes as on reading the data.

### 6.1 What was implemented

`cuda_impl::SliceTiles` in `histogram.cu`, declared in `histogram.cuh` so it can be unit
tested. Three bounds with an explicit priority:

```cpp
auto n_tiles = DivRoundUp(n_items, tile_size);
// flush_bytes / entry_bytes <= kMaxFlushPercent / 100, with
//   flush_bytes = n_targets * shmem_bytes
//   entry_bytes = n_tiles_per_blk * tile_size * entry_bits / 8
auto tiles_flush  = DivRoundUp(100 * 8 * n_targets * shmem_bytes,
                               kMaxFlushPercent * tile_size * entry_bits);
auto tiles_target = DivRoundUp(n_tiles, n_resident_blks_per_target * kTargetWaves);
auto tiles_cap    = DivRoundUp(n_tiles, n_resident_blks_per_target * kMinWaves);

auto n_tiles_per_blk = std::max(1, std::min(std::max(tiles_target, tiles_flush), tiles_cap));
```

`shmem_bytes` is `feature_groups.ShmemSize()`, and is zero on the global-memory path, where
there is no flush and hence no floor. `entry_bits` comes from the new `EntryBits(matrix)`,
which recovers the symbol count from the accessor (`NullValue()` is the symbol count for a
fully dense page, one less otherwise) and takes `common::detail::SymbolBits` of it — so a wider
gradient index correctly makes the flush relatively cheaper.

`HistKernel::SliceItems` is now a thin wrapper that converts tiles to entries.

### 6.2 Effect

512 features, 4 targets, 188 SMs:

| rows per launch | before t/blk | waves | flush | after t/blk | waves | flush |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 M | 30 | 1.0 | 107 % | 8 | 3.6 | 400 % |
| 0.26 M | 32 | 7.3 | **100 %** | 59 | 3.9 | **54 %** |
| 0.52 M | 32 | 14.5 | **100 %** | 117 | 4.0 | **27 %** |
| 1.05 M | 32 | 29.1 | **100 %** | 128 | 7.3 | **25 %** |
| 4.19 M | 117 | 31.8 | 27 % | 128 | 29.1 | 25 % |
| 16.78 M | 465 | 32.0 | 7 % | 465 | 32.0 | 7 % |
| 33.55 M | 930 | 32.0 | 3 % | 930 | 32.0 | 3 % |

Large in-core launches are unchanged — the floor never binds against the target. Very small
launches (first row) now get more waves and more flush: `kMinWaves` is a cap on chunk length, so
it can only raise the block count. At that size the absolute volumes are tiny (16 MB of
gradient index), and the utilisation gain is real, but it is a behaviour change.

### 6.3 Verification performed

- `testxgboost --gtest_filter='*Histogram*:*GpuHist*:*GPUHist*:*Ellpack*:*Driver*:*QuantileDMatrix*'`
  — 244 passed, 3 pre-existing skips.
- `pytest tests/python-gpu/test_gpu_updaters.py` — 30 passed.
  `test_gpu_data_iterator.py`, `test_gpu_prediction.py` — 46 passed, 3 failures that reproduce
  on the un-patched branch build (`test_predict_leaf_basic`, unrelated).
- **Bitwise identical results**, in-core and external memory: the per-round RMSE sequences match
  the un-patched build exactly, as expected from integer atomics.
- Local 46 SM sm_120 part, external memory, 4 × 1 M-row pages, 256 features, 4 targets: the
  policy is active (histogram grid 2588 → 648 blocks for node levels, 2916 → 1368 for the root,
  i.e. **4× less flush**) and the total histogram kernel time is **unchanged**: 1.097 s vs
  1.101 s over 90 launches. Whole-training wall time differs by less than the noise of this
  power-capped part.

So locally the trade is free: 4× less flush traffic for the same kernel time. The regime this
is aimed at — enough waves that the reduced block count costs nothing — needs the RTX PRO 6000
to confirm.

### 6.4 Gate before Phase 2

External-memory benchmarks (`ext-qdm-iter`, varying `--n_batches`) on the RTX PRO 6000 must
improve or be neutral, and in-core timings must be unchanged. Also sweep `kMaxFlushPercent`
there, since it and `kMinWaves` are the two declared budgets and small pages are where they
bind.

## 7. Phase 2 — decomposition and ordering — IMPLEMENTED

### 7.1 Work unit and ordering

A block accumulates one chunk of one `(node, feature group)` segment into one target. Chunks
are `n_entries_per_chunk` **entries** inside a segment, so no block spans two segments. The
block index decodes as

```
((nidx_in_set * n_chunks_per_segment + chunk) * n_groups + group) * n_targets + target
```

— target fastest, group next. The grid is rectangular and covers the largest segment of the
launch, so uneven segments leave empty blocks, which return before touching shared memory.

```cpp
std::uint32_t blk = blockIdx.x;
bst_target_t const target_idx = blk % n_targets;       blk /= n_targets;
bst_feature_t const gidx = blk % n_groups;             blk /= n_groups;
std::uint32_t const chunk_idx = blk % n_chunks_per_segment;
std::size_t const nidx_in_set = blk / n_chunks_per_segment;

bst_feature_t const feature_stride = Policy::kCompressed ? group.num_features : matrix.row_stride;
bst_idx_t const n_entries = d_ridx.size() * feature_stride;
bst_idx_t const begin = chunk_idx * n_entries_per_chunk;
if (begin >= n_entries) { return; }   // the grid covers the largest segment
bst_idx_t const end = min(begin + n_entries_per_chunk, n_entries);
```

`HistKernelSegment` is reused unchanged: it takes the full row-index span with segment-local
`begin`/`end` and derives `ridx_in_set = idx / feature_stride` itself.

Chunking in entries rather than rows is deliberate. `FeatureGroups` packs features until the bin
budget is full, so uneven per-feature bin counts give groups of very different widths. With
entry chunks the work of a non-empty block is `n_entries_per_chunk` in every group, so the width
skew creates no tail. The cost is that front alignment is exact only within a group-width class,
which is where it matters: wide groups (`>= kSectorBytes / bytes_per_entry` features) fill
sectors alone, and narrow groups are the bin-limited ones, which share a width.

### 7.2 Grid shape

`cuda_impl::MakeChunkGrid`, declared in `histogram.cuh` so it can be unit tested, returns
`{n_entries_per_chunk, n_chunks_per_segment, n_blks}` and lengthens the chunk rather than
overflowing a 32-bit grid:

```cpp
auto n_blks_per_chunk = n_groups * n_targets * n_nodes;
CHECK_LE(n_blks_per_chunk, kMaxGrid);
auto n_chunks = DivRoundUp(max_segment_entries, n_entries_per_chunk);
while (n_chunks > kMaxGrid / n_blks_per_chunk) {   // longer chunks for fewer blocks
  n_entries_per_chunk *= 2;
  n_chunks = DivRoundUp(max_segment_entries, n_entries_per_chunk);
}
```

Host side, `max_segment_entries = max_node_rows * max_group_features`, both derived from data
already present: node row counts from `h_ridx_iters`, group widths from
`h_feature_groups.feature_segments` (the sparse layout has a single group spanning the row).

### 7.3 Invariants obtained

| quantity | before | after |
|---|---|---|
| co-resident feature groups | `min(n_groups, n_resident_chunks * chunk / segment_g)` — 1.3 at level 1, 42 at level 6, and varying per group under width skew | `min(n_groups, n_resident_blocks / n_targets)` — **independent of chunk size and of depth** |
| work per non-empty block | `n_items_per_blk` entries, chunk may span segments | exactly `n_entries_per_chunk` entries, one segment |
| device arrays per launch | `ridx_iters`, `hists`, `sizes_csum` (40 KB at `n_nodes = 1024`) | `ridx_iters`, `hists` (**32 KB**) |
| flushes per block | 1 per segment visited | exactly 1 |

### 7.4 Removed

`HistSegment`, `FindSegment`, `UpperBoundIdx` (its only user), the `while (pos < last())` loop,
the post-barrier `find_segment` recompute, `volatile bst_idx_t pos`, the `sizes_csum` kernel
parameter and its `TemporaryArray`/`dh::CopyTo`, and `kItemsPerThread` with
`HistPolicy::kTileSize`. `SliceTiles` is now called with `Policy::kBlockThreads` as the tile
size — one block-stride pass — which leaves `n_entries_per_chunk` numerically unchanged, since
both the flush floor and the load-balance target scale with the unit.

### 7.5 Unchanged

`FeatureGroups` and `feature_groups.cu`, `HistShmemBytes`, `Dft{St,Mt}HistShmemBytes`,
`HistTuning` and the launch bounds, `BlocksPerMp`, `DispatchCudaSm`, `HistKernelSegment`,
`IterIdx`, `LoadGpair`, the atomics, `DeviceHistogramStorage`, `AllReduceHist`,
`SubtractionTrick`, both `BuildHistogram` overloads, and both call sites in
`updater_gpu_hist.cu{,h}`.

### 7.6 Verification performed

- `testxgboost --gtest_filter='*Histogram*:*GpuHist*:*GPUHist*:*Ellpack*:*Driver*:*QuantileDMatrix*'`
  — **246 passed**, 3 pre-existing skips. The pre-existing `HistogramBuildTest` matrix
  (dense / dense-missing / sparse × 1,3 targets × root/nodes × shared/global × `small_groups`)
  passes unmodified, including its uneven node sizes `{0, 1, 7, 0, 1000, …}`, which exercise
  empty rectangular slots and segments shorter than one chunk.
- **Bitwise identical results** versus the Phase 1 build, in-core and external memory
  (`rmse` sequences match to all digits).
- Registers and local memory, `regcheck.sh` with `scratch/spills.py` (attributes every
  `LDL`/`STL` to its innermost enclosing backward branch via `nvdisasm -g`). sm_120 is
  **completely clean**. On sm_80 and sm_90 all spilling instantiations are the
  **`DoubleEllpackAccessor`** ones (`Single = 0/48` on every arch), which sit at the
  32-register cap of their `(1024, 2)` launch bounds; `DoubleCompressedIter` carries two
  buffer pointers where `CompressedIterator` carries one. Measured as local-memory ops
  **inside** the accumulation loop — which is what costs time:

  | in-loop local-memory ops | Phase 1 | Phase 2 |
  |---|---:|---:|
  | sm_90, accumulation loop | 80 | 74 |
  | sm_90, outer `while (pos < last())` loop | **902** | **0 (loop removed)** |
  | sm_90 total | 982 | **74** |
  | sm_80 total | 667 | **38** |
  | instantiations with in-loop local traffic | 96/96, both accessors | 27/96, double only |

  The 902 ops were the deliberate `volatile bst_idx_t pos`, not spills, but real traffic all
  the same. So Phase 2 is 13× better on sm_90 and 17× better on sm_80, and clean for the
  single accessor. The residual 2–4 ops per iteration are Phase 5 below.
- Local 46 SM sm_120 part, 4 M rows, 256 features, 4 targets, depthwise — histogram kernel
  time from the nsys trace, and whole-training time over 3 interleaved repetitions:

  | | hist kernel | train (min of 3) | train (median) |
  |---|---:|---:|---:|
  | in-core, Phase 1 | 2.306 s | 3.555 s | 3.749 s |
  | in-core, Phase 2 | **1.854 s (−19.6 %)** | **2.899 s** | **2.951 s** |
  | extmem, Phase 1 | 2.188 s | 3.560 s | 3.601 s |
  | extmem, Phase 2 | **1.833 s (−16.2 %)** | **2.977 s** | **2.998 s** |

  Grid sizes are comparable (node levels 2584 → 2728 in-core, 648 → 704 external memory), so
  the gain is not from fewer blocks. On this part `n_resident_blocks / n_targets = 23` against
  `n_groups = 22`, so group co-residency goes from ~1.3 at shallow levels to all 22 groups at
  every level, which is the predicted mechanism.

### 7.7 Gate before Phase 3

The four regressing configurations on the RTX PRO 6000, against master and against the
pre-Phase-1 branch. The sharp prediction to check is 8.4: the optimum `n_entries_per_chunk`
should **stop moving with tree depth**.

## 8. Phase 3 — measurement gate

All variants are bitwise identical (integer atomics), so sweeps carry no correctness risk.

1. **Correctness:** `testxgboost --gtest_filter=*Histogram*` plus `tests/python-gpu`.
2. **Registers:** `./regcheck.sh {75,80,86,90,100,120}` with `scratch/spills.py`. Expect the
   8-byte stack frame gone, sm_120 clean, and no *new* in-loop local-memory traffic beyond the
   double-accessor residual recorded in 7.6. Done; see 7.6.
3. **In-core, the full matrix:** all eight configurations of
   `benchmark-training-comparison.md` (256/512 features × 1/4 targets × depthwise/lossguide),
   versus master and versus the branch. **Not yet done, even locally** — only
   256 features / 4 targets / depthwise has been measured. The single-target and lossguide rows
   are the ones that must not regress, so they are the point of the exercise.
   `scratch/matrix.sh` runs the matrix but needs a fix first: it replaces
   `libxgboost.so` with `cp` while the file may still be mapped, which gives a `SIGBUS`.
   Copy to a temporary path and `mv` (atomic rename, new inode) instead.
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

1. **Skewed group widths** — *added in Phase 2*: `HistInput` takes a `skewed` flag giving the
   first half of the features `n_bins` and the second half `n_bins / 32`, in contiguous runs so
   the groups do not mix the two and even out. `Histogram.BuildSkewedGroups` covers
   dense / dense-missing / sparse × 1,3 targets × shared/global and asserts
   `max_group_features > 2 * min_group_features`. `HistInput` now carries per-feature bin counts
   throughout (`feature_bins`, `bin_ptrs`), so `MakeEllpack` and `Expected` handle uneven cuts.
2. **Empty-slot coverage.** Deliberately uneven node sizes (e.g. `{1, 1024, 7, 65536}`) so many
   rectangular slots are empty, including a node with fewer rows than `n_chunks_per_segment`.
3. **Chunk-boundary coverage.** Covered by the existing `HistogramBuildTest` node sizes
   `{0, 1, 7, 0, 1000, …}` (segments far shorter than a chunk, hence the `begin >= n_entries`
   path) together with `Histogram.BuildLarge` (several chunks per segment).
4. **Grid guard** — *added in Phase 2*: `Histogram.MakeChunkGrid` pins the even case, the
   chunk-longer-than-segment case, and the overflow case (512 groups × 32 targets × 1024 nodes
   over a 2^40-entry segment), asserting the chunk is lengthened, the grid fits, and the chunks
   still cover the largest segment.
5. **Derived floor** — *added in Phase 1*: `Histogram.SliceTilesFlushFloor` asserts the floor is
   proportional to `n_targets`, to `ShmemSize()`, and inversely proportional to `entry_bits`, and
   that the declared budget is met without being over-spent.
6. **Bound priority** — *added in Phase 1*: `Histogram.SliceTilesBounds` pins all three bounds —
   the target binds on large launches, the `kMinWaves` cap wins over the flush floor on small
   ones, the global-memory path has no floor, and the result is never zero.

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
| sm_80 / sm_90 keep 2–4 in-loop local-memory ops in the double-accessor instantiations | 13–17× less in-loop local traffic than Phase 1, and sm_120 is clean; addressed by Phase 5 |

Phases 1 and 2 are separate commits with independent gates. Phase 1 is a ~10-line change to one
function and is reversible on its own. Phase 2 is confined to `HistogramKernel` and
`DispatchHist` and leaves every public signature and every other file unchanged.

## 13. Phase 5 — double-accessor register pressure (optional, independent)

Every remaining spill is in the `DoubleEllpackAccessor` instantiations on sm_80 and sm_90, 2–4
local-memory ops per iteration of the accumulation loop. Cause: `(1024, 2)` launch bounds cap
registers at `65536 / (1024 * 2) = 32`, and `DoubleCompressedIter` needs more live state than
`CompressedIterator`. sm_120's `(768, 2)` gives a 42-register cap and is clean.

Two independent options:

- Give the double-accessor instantiations their own launch bounds. `(1024, 1)` on sm_80 and
  sm_90 raises the cap to 64 registers. The double accessor is the external-memory path, where
  the page fetch dominates and occupancy matters less.
- Shrink the live state of `DoubleCompressedIter::operator[]`.

Independent of Phases 3 and 4, and measurable with `scratch/spills.py` plus an external-memory
benchmark. Not required for the regression this plan addresses.

## 14. Out of scope

- `targets_per_block > 1` (one block owning several targets' histograms, removing the target
  axis entirely at the cost of narrowing `features_per_group`).
- Changes to `FeatureGroups`, including a `max_features_per_group` cap.
- Making `use_shared` a profitability decision rather than a "does it fit" test.
- The `n_targets` ≳ 24 regime, where the privatized histogram stops paying for itself.
