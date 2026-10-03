/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // for Context

#include <algorithm>  // for shuffle, none_of, transform
#include <limits>     // for numeric_limits
#include <memory>     // for unique_ptr
#include <numeric>    // for iota, accumulate
#include <random>     // for mt19937
#include <sstream>    // for stringstream
#include <tuple>      // for tuple
#include <vector>     // for vector

#include "../../../../src/common/cuda_rt_utils.h"          // for GetMpCnt
#include "../../../../src/data/device_adapter.cuh"         // for CupyAdapter, GetRowCounts
#include "../../../../src/tree/gpu_hist/expand_entry.cuh"  // for GPUExpandEntry
#include "../../../../src/tree/gpu_hist/histogram.cuh"
#include "../../../../src/tree/gpu_hist/row_partitioner.cuh"  // for RowPartitioner
#include "../../../../src/tree/hist/hist_param.h"             // for HistMakerTrainParam
#include "../../../../src/tree/param.h"                       // for TrainParam
#include "../../categorical_helpers.h"                        // for OneHotEncodeFeature
#include "../../helpers.h"
#include "dummy_quantizer.cuh"

namespace xgboost::tree {
TEST(Histogram, HistShmemBytes) {
  auto device = 0;
  auto optin = dh::MaxSharedMemoryOptin(device);
  for (auto budget : {DftStHistShmemBytes(device), DftMtHistShmemBytes(device)}) {
    ASSERT_GT(budget, 0);
    ASSERT_LE(budget, optin);
  }
}

namespace {
// Mirrors the constants of `cuda_impl::SliceTiles`.
constexpr std::size_t kTargetWaves = 32;
constexpr std::size_t kMinWaves = 4;
constexpr std::size_t kMaxFlushPercent = 25;

// The flush volume implied by a slice, as a fraction of the gradient index bytes read.
[[nodiscard]] double FlushRatio(std::size_t n_tiles_per_blk, std::size_t tile_size,
                                bst_target_t n_targets, std::size_t shmem_bytes,
                                std::uint32_t entry_bits) {
  auto flush = static_cast<double>(n_targets) * static_cast<double>(shmem_bytes);
  auto entries = static_cast<double>(n_tiles_per_blk) * static_cast<double>(tile_size) *
                 static_cast<double>(entry_bits) / 8.0;
  return flush / entries;
}
}  // anonymous namespace

// The flush floor scales with the number of targets and the size of the privatized
// histogram. The constant it replaced did not.
TEST(Histogram, SliceTilesFlushFloor) {
  std::size_t constexpr kTileSize = 768 * 8;
  std::size_t constexpr kShmem = 12 * 256 * sizeof(GradientPairInt64);  // 49152
  std::uint32_t constexpr kEntryBits = 8;
  std::size_t constexpr kResident = 32;

  for (bst_target_t n_targets : {1, 2, 4, 8, 32}) {
    // Sized so that the flush floor is the binding bound: above the load-balance target and
    // below the one-wave cap.
    auto n_tiles = kResident * kTargetWaves * 16 * n_targets;
    auto n_items = static_cast<bst_idx_t>(n_tiles) * kTileSize;
    auto [tiles, blks] =
        cuda_impl::SliceTiles(n_items, kTileSize, kResident, n_targets, kShmem, kEntryBits);

    // 100 * 8 * n_targets * kShmem / (kMaxFlushPercent * kTileSize * kEntryBits)
    ASSERT_EQ(tiles, 32u * n_targets) << "n_targets:" << n_targets;
    ASSERT_EQ(blks, common::DivRoundUp(n_tiles, tiles));
    // The declared budget is met, and not over-spent.
    auto ratio = FlushRatio(tiles, kTileSize, n_targets, kShmem, kEntryBits);
    ASSERT_LE(ratio, static_cast<double>(kMaxFlushPercent) / 100.0 + 1e-9);
    ASSERT_GT(ratio, static_cast<double>(kMaxFlushPercent) / 100.0 * 0.5);
  }

  // Proportional to the size of the privatized histogram as well. Sized so the floor binds
  // for every variant below: above the load-balance target, below the one-wave cap.
  auto n_items = static_cast<bst_idx_t>(kResident * kTargetWaves * 8) * kTileSize;
  auto [small, _s] = cuda_impl::SliceTiles(n_items, kTileSize, kResident, 1, kShmem, kEntryBits);
  auto [large, _l] =
      cuda_impl::SliceTiles(n_items, kTileSize, kResident, 1, kShmem * 2, kEntryBits);
  ASSERT_EQ(large, small * 2);

  // A wider gradient index carries more bytes per entry, so the flush is relatively cheaper.
  auto [narrow, _n] = cuda_impl::SliceTiles(n_items, kTileSize, kResident, 1, kShmem, 8);
  auto [wide, _w] = cuda_impl::SliceTiles(n_items, kTileSize, kResident, 1, kShmem, 16);
  ASSERT_EQ(narrow, wide * 2);
}

// The three bounds and their priority.
TEST(Histogram, SliceTilesBounds) {
  std::size_t constexpr kTileSize = 768 * 8;
  std::size_t constexpr kShmem = 12 * 256 * sizeof(GradientPairInt64);
  std::uint32_t constexpr kEntryBits = 8;
  bst_target_t constexpr kTargets = 4;
  std::size_t constexpr kResident = 94;

  {
    // Large launch: the load-balance target binds, giving exactly kTargetWaves waves.
    auto n_tiles = kResident * kTargetWaves * 1024;
    auto n_items = static_cast<bst_idx_t>(n_tiles) * kTileSize;
    auto [tiles, blks] =
        cuda_impl::SliceTiles(n_items, kTileSize, kResident, kTargets, kShmem, kEntryBits);
    ASSERT_EQ(tiles, common::DivRoundUp(n_tiles, kResident * kTargetWaves));
    ASSERT_EQ(blks, kResident * kTargetWaves);
  }
  {
    // Small launch: the flush floor would ask for longer blocks than kMinWaves allows, so
    // the cap wins and the tail stays bounded.
    auto n_tiles = kResident * kMinWaves * 8;
    auto n_items = static_cast<bst_idx_t>(n_tiles) * kTileSize;
    auto [tiles, blks] =
        cuda_impl::SliceTiles(n_items, kTileSize, kResident, kTargets, kShmem, kEntryBits);
    ASSERT_EQ(tiles, common::DivRoundUp(n_tiles, kResident * kMinWaves));
    ASSERT_EQ(blks, kResident * kMinWaves);
    // The budget cannot be met at this size; the cap is what limits it.
    ASSERT_GT(FlushRatio(tiles, kTileSize, kTargets, kShmem, kEntryBits),
              static_cast<double>(kMaxFlushPercent) / 100.0);
  }
  {
    // Global memory: no privatized histogram, so no flush and no floor.
    auto n_tiles = kResident * kTargetWaves * 8;
    auto n_items = static_cast<bst_idx_t>(n_tiles) * kTileSize;
    auto [tiles, blks] = cuda_impl::SliceTiles(n_items, kTileSize, kResident, kTargets,
                                               /*shmem_bytes=*/0, kEntryBits);
    ASSERT_EQ(tiles, common::DivRoundUp(n_tiles, kResident * kTargetWaves));
    ASSERT_EQ(blks, kResident * kTargetWaves);
  }
  {
    // Never zero.
    auto [tiles, blks] = cuda_impl::SliceTiles(1, kTileSize, 1, 1, kShmem, kEntryBits);
    ASSERT_EQ(tiles, 1u);
    ASSERT_EQ(blks, 1u);
  }
}

// The grid is rectangular over (node, feature group, chunk, target), and lengthens the chunk
// rather than overflowing.
TEST(Histogram, MakeChunkGrid) {
  {
    // Even segments: no empty block.
    auto grid = cuda_impl::MakeChunkGrid(/*max_segment_entries=*/1024, /*chunk=*/256,
                                         /*n_groups=*/4, /*n_targets=*/2, /*n_nodes=*/3);
    ASSERT_EQ(grid.n_entries_per_chunk, 256u);
    ASSERT_EQ(grid.n_chunks_per_segment, 4u);
    ASSERT_EQ(grid.n_blks, 4u * 4u * 2u * 3u);
  }
  {
    // A chunk longer than the largest segment still gets one block per segment.
    auto grid = cuda_impl::MakeChunkGrid(10, 4096, 1, 1, 7);
    ASSERT_EQ(grid.n_chunks_per_segment, 1u);
    ASSERT_EQ(grid.n_blks, 7u);
  }
  {
    // The grid would overflow, so the chunk is lengthened instead.
    std::uint32_t constexpr kGroups = 512;
    bst_target_t constexpr kTargets = 32;
    std::size_t constexpr kNodes = 1024;
    bst_idx_t constexpr kSegment = bst_idx_t{1} << 40;
    auto grid = cuda_impl::MakeChunkGrid(kSegment, /*chunk=*/1024, kGroups, kTargets, kNodes);
    ASSERT_GT(grid.n_entries_per_chunk, 1024u);
    ASSERT_EQ(grid.n_chunks_per_segment, common::DivRoundUp(kSegment, grid.n_entries_per_chunk));
    // Fits, and covers the largest segment.
    ASSERT_LE(static_cast<std::uint64_t>(grid.n_blks), std::numeric_limits<std::uint32_t>::max());
    ASSERT_GE(static_cast<bst_idx_t>(grid.n_chunks_per_segment) * grid.n_entries_per_chunk,
              kSegment);
  }
}

TEST(Histogram, DeviceHistogramStorage) {
  // Ensures that node allocates correctly after reaching `kStopGrowingSize`.
  auto ctx = MakeCUDACtx(0);
  constexpr size_t kNBins = 128;
  constexpr int kNNodes = 4;
  constexpr size_t kStopGrowing = kNNodes * kNBins * 2u;
  DeviceHistogramStorage histogram{};
  histogram.Reset(&ctx, kNBins, kNNodes);
  for (int i = 0; i < kNNodes; ++i) {
    histogram.AllocateHistograms(&ctx, {i});
  }
  ASSERT_EQ(histogram.Data().size(), kStopGrowing);
  histogram.Reset(&ctx, kNBins, kNNodes);

  // Use allocated memory but do not erase nidx_map.
  for (int i = 0; i < kNNodes; ++i) {
    histogram.AllocateHistograms(&ctx, {i});
  }
  for (int i = 0; i < kNNodes; ++i) {
    ASSERT_TRUE(histogram.HistogramExists(i));
  }

  // Add two new nodes
  histogram.AllocateHistograms(&ctx, {kNNodes});
  histogram.AllocateHistograms(&ctx, {kNNodes + 1});

  // Old cached nodes should still exist
  for (int i = 0; i < kNNodes; ++i) {
    ASSERT_TRUE(histogram.HistogramExists(i));
  }

  // Should be deleted
  ASSERT_FALSE(histogram.HistogramExists(kNNodes));
  // Most recent node should exist
  ASSERT_TRUE(histogram.HistogramExists(kNNodes + 1));

  // Add same node again - should fail
  EXPECT_ANY_THROW(histogram.AllocateHistograms(&ctx, {kNNodes + 1}););
}

TEST(Histogram, SubtractionTrick) {
  auto ctx = MakeCUDACtx(0);
  bst_bin_t n_bins = 16;

  DeviceHistogramBuilder histogram;
  // Only the root is cached, the other nodes are in the overflow buffer.
  histogram.Reset(&ctx, /*max_cached_hist_nodes=*/1, n_bins, false);
  histogram.AllocateHistograms(&ctx, {0});
  histogram.AllocateHistograms(&ctx, {1}, {2});

  auto fill = [&](bst_node_t nidx, GradientPairInt64 v) {
    auto hist = histogram.GetNodeHistogram(nidx);
    thrust::fill(ctx.CUDACtx()->CTP(), dh::tbegin(hist), dh::tend(hist), v);
  };
  fill(0, GradientPairInt64{10, 20});
  fill(1, GradientPairInt64{3, 4});

  GPUExpandEntry root;
  root.nidx = 0;
  auto need_build = histogram.SubtractHist<GPUExpandEntry>(&ctx, {root}, {1}, {2});
  ASSERT_TRUE(need_build.empty());
  std::vector<GradientPairInt64> h_hist(n_bins);
  dh::CopyDeviceSpanToVector(&h_hist, histogram.GetNodeHistogram(2));
  for (auto v : h_hist) {
    ASSERT_EQ(v, (GradientPairInt64{7, 16}));
  }

  // Allocating the next level clears the overflow buffer, the parents are no longer available.
  histogram.AllocateHistograms(&ctx, {3, 5}, {4, 6});
  std::vector<GPUExpandEntry> candidates(2);
  candidates[0].nidx = 1;
  candidates[1].nidx = 2;
  need_build = histogram.SubtractHist(&ctx, candidates, {3, 5}, {4, 6});
  ASSERT_EQ(need_build, (std::vector<bst_node_t>{4, 6}));
}

void ValidateCategoricalHistogram(size_t n_categories, common::Span<GradientPairInt64> onehot,
                                  common::Span<GradientPairInt64> cat) {
  auto cat_sum = std::accumulate(cat.cbegin(), cat.cend(), GradientPairInt64{});
  for (size_t c = 0; c < n_categories; ++c) {
    auto zero = onehot[c * 2];
    auto one = onehot[c * 2 + 1];

    auto chosen = cat[c];
    auto not_chosen = cat_sum - chosen;
    ASSERT_EQ(zero, not_chosen);
    ASSERT_EQ(one, chosen);
  }
}

// Test 1 vs rest categorical histogram is equivalent to one hot encoded data.
void TestGPUHistogramCategorical(size_t num_categories) {
  auto ctx = MakeCUDACtx(0);
  size_t kRows = std::max(static_cast<decltype(num_categories)>(340), num_categories);
  size_t constexpr kBins = 256;
  auto x = GenerateRandomCategoricalSingleColumn(kRows, num_categories);
  auto cat_m = GetDMatrixFromData(x, kRows, 1);
  cat_m->Info().feature_types.HostVector().push_back(FeatureType::kCategorical);
  auto batch_param = BatchParam{kBins, tree::TrainParam::DftSparseThreshold()};
  tree::RowPartitioner row_partitioner;
  row_partitioner.Reset(&ctx, kRows, 0);
  auto ridx = row_partitioner.GetRows(0);
  dh::device_vector<GradientPairInt64> cat_hist(num_categories);

  auto gpairs_i64 = GenerateGradientsFixedPoint(&ctx, kRows, 1, 0.0f, 2.0f).gpair;
  /**
   * Generate hist with cat data.
   */
  for (auto const& batch : cat_m->GetBatches<EllpackPage>(&ctx, batch_param)) {
    auto* page = batch.Impl();
    FeatureGroups single_group(page->Cuts());
    DeviceHistogramBuilder builder;
    builder.Reset(&ctx, HistMakerTrainParam::CudaDefaultNodes(), num_categories, false);
    page->Visit(&ctx, {}, [&](auto&& acc) {
      builder.BuildHistogram(&ctx, acc, single_group, gpairs_i64.View(ctx.Device()).Values(), ridx,
                             dh::ToSpan(cat_hist));
    });
  }

  /**
   * Generate hist with one hot encoded data.
   */
  auto x_encoded = OneHotEncodeFeature(x, num_categories);
  auto encode_m = GetDMatrixFromData(x_encoded, kRows, num_categories);
  dh::device_vector<GradientPairInt64> encode_hist(2 * num_categories);
  for (auto const& batch : encode_m->GetBatches<EllpackPage>(&ctx, batch_param)) {
    auto* page = batch.Impl();
    FeatureGroups single_group(page->Cuts());
    DeviceHistogramBuilder builder;
    builder.Reset(&ctx, HistMakerTrainParam::CudaDefaultNodes(), encode_hist.size(), false);
    page->Visit(&ctx, {}, [&](auto&& acc) {
      builder.BuildHistogram(&ctx, acc, single_group, gpairs_i64.View(ctx.Device()).Values(), ridx,
                             dh::ToSpan(encode_hist));
    });
  }

  std::vector<GradientPairInt64> h_cat_hist(cat_hist.size());
  thrust::copy(cat_hist.begin(), cat_hist.end(), h_cat_hist.begin());

  std::vector<GradientPairInt64> h_encode_hist(encode_hist.size());
  thrust::copy(encode_hist.begin(), encode_hist.end(), h_encode_hist.begin());
  ValidateCategoricalHistogram(num_categories, common::Span<GradientPairInt64>{h_encode_hist},
                               common::Span<GradientPairInt64>{h_cat_hist});
}

TEST(Histogram, GPUHistCategorical) {
  for (size_t num_categories = 2; num_categories < 8; ++num_categories) {
    TestGPUHistogramCategorical(num_categories);
  }
  // Larger than the shared memory size, must use global memory since there's no feature
  // group with a single feature.
  auto max_shmem = dh::MaxSharedMemoryOptin(0);
  auto n_categories = common::DivRoundUp(max_shmem, sizeof(GradientPairInt64)) * 2;
  TestGPUHistogramCategorical(n_categories);
}

namespace {
// Atomic add as type cast for test.
XGBOOST_DEV_INLINE int64_t atomicAdd(int64_t* dst, int64_t src) {  // NOLINT
  uint64_t* u_dst = reinterpret_cast<uint64_t*>(dst);
  uint64_t u_src = *reinterpret_cast<uint64_t*>(&src);
  uint64_t ret = ::atomicAdd(u_dst, u_src);
  return *reinterpret_cast<int64_t*>(&ret);
}
}  // namespace

void TestAtomicAdd() {
  size_t n_elements = 1024;
  dh::device_vector<int64_t> result_a(1, 0);
  auto d_result_a = result_a.data().get();

  dh::device_vector<int64_t> result_b(1, 0);
  auto d_result_b = result_b.data().get();

  /**
   * Test for simple inputs
   */
  std::vector<int64_t> h_inputs(n_elements);
  for (size_t i = 0; i < h_inputs.size(); ++i) {
    h_inputs[i] = (i % 2 == 0) ? i : -i;
  }
  dh::device_vector<int64_t> inputs(h_inputs);
  auto d_inputs = inputs.data().get();

  dh::LaunchN(n_elements, [=] __device__(size_t i) {
    AtomicAdd64As32(d_result_a, d_inputs[i]);
    atomicAdd(d_result_b, d_inputs[i]);
  });
  ASSERT_EQ(result_a[0], result_b[0]);

  /**
   * Test for positive values that don't fit into 32 bit integer.
   */
  thrust::fill(inputs.begin(), inputs.end(), (std::numeric_limits<uint32_t>::max() / 2));
  thrust::fill(result_a.begin(), result_a.end(), 0);
  thrust::fill(result_b.begin(), result_b.end(), 0);
  dh::LaunchN(n_elements, [=] __device__(size_t i) {
    AtomicAdd64As32(d_result_a, d_inputs[i]);
    atomicAdd(d_result_b, d_inputs[i]);
  });
  ASSERT_EQ(result_a[0], result_b[0]);
  ASSERT_GT(result_a[0], std::numeric_limits<uint32_t>::max());
  CHECK_EQ(thrust::reduce(inputs.begin(), inputs.end(), int64_t(0)), result_a[0]);

  /**
   * Test for negative values that don't fit into 32 bit integer.
   */
  thrust::fill(inputs.begin(), inputs.end(), (std::numeric_limits<int32_t>::min() / 2));
  thrust::fill(result_a.begin(), result_a.end(), 0);
  thrust::fill(result_b.begin(), result_b.end(), 0);
  dh::LaunchN(n_elements, [=] __device__(size_t i) {
    AtomicAdd64As32(d_result_a, d_inputs[i]);
    atomicAdd(d_result_b, d_inputs[i]);
  });
  ASSERT_EQ(result_a[0], result_b[0]);
  ASSERT_LT(result_a[0], std::numeric_limits<int32_t>::min());
  CHECK_EQ(thrust::reduce(inputs.begin(), inputs.end(), int64_t(0)), result_a[0]);
}

TEST(Histogram, AtomicAddInt64) { TestAtomicAdd(); }

TEST(Histogram, Quantiser) {
  auto ctx = MakeCUDACtx(0);
  std::size_t n_samples{16};
  HostDeviceVector<GradientPair> gpair(n_samples, GradientPair{1.0, 1.0});
  gpair.SetDevice(ctx.Device());

  GradientQuantiserGroup quantiser_group{&ctx,
                                         linalg::MakeVec(ctx.Device(), gpair.ConstDeviceSpan())};
  auto quantiser = quantiser_group[0];
  for (auto v : gpair.ConstHostVector()) {
    auto gh = quantiser.ToFloatingPoint(quantiser.ToFixedPoint(v));
    ASSERT_EQ(gh.GetGrad(), 1.0);
    ASSERT_EQ(gh.GetHess(), 1.0);
  }

  GradientQuantiser hess_floor_quantiser{GradientPairPrecise{1.0, 10.0},
                                         GradientPairPrecise{1.0, 0.1}};
  auto tiny_hess = hess_floor_quantiser.ToFixedPoint(GradientPairPrecise{0.25, 0.05});
  ASSERT_EQ(tiny_hess.GetQuantisedGrad(), 0);
  ASSERT_EQ(tiny_hess.GetQuantisedHess(), 1);

  auto zero_hess = hess_floor_quantiser.ToFixedPoint(GradientPairPrecise{0.25, 0.0});
  ASSERT_EQ(zero_hess.GetQuantisedHess(), 0);

  auto tiny_hess_float = hess_floor_quantiser.ToFixedPoint(GradientPair{0.25f, 0.05f});
  ASSERT_EQ(tiny_hess_float.GetQuantisedGrad(), 0);
  ASSERT_EQ(tiny_hess_float.GetQuantisedHess(), 1);
}
namespace {
enum CacheMode {
  kNoCache = 0,
  kCopy = 1,
  kDirect = 2,
};

class HistogramExternalMemoryTest
    : public ::testing::TestWithParam<std::tuple<float, bool, CacheMode>> {
 public:
  void Run(float sparsity, bool force_global, CacheMode cache_mode) {
    auto ctx = MakeCUDACtx(0);
    bst_idx_t n_samples{512}, n_features{12}, n_batches{3};
    std::vector<std::unique_ptr<RowPartitioner>> partitioners;
    auto rng = RandomDataGenerator{n_samples, n_features, sparsity}.Batches(n_batches);
    bst_bin_t n_bins = 16;
    std::shared_ptr<DMatrix> p_fmat;
    switch (cache_mode) {
      case kCopy:
      case kDirect: {
        p_fmat = rng.CacheHostRatio(0.5)
                     .Device(ctx.Device())
                     .Bins(n_bins)
                     .OnHost(true)
                     .MinPageCacheBytes(n_bins * n_features)
                     .GenerateExtMemQuantileDMatrix("cache", true);
        break;
      }
      case kNoCache: {
        p_fmat = rng.GenerateSparsePageDMatrix("cache", true);
        break;
      }
    }

    BatchParam p{n_bins, TrainParam::DftSparseThreshold()};
    if (cache_mode == kDirect) {
      p.prefetch_copy = false;
    } else if (cache_mode == kCopy) {
      p.prefetch_copy = true;
    }

    std::unique_ptr<FeatureGroups> fg;
    dh::device_vector<GradientPairInt64> single_hist;
    dh::device_vector<GradientPairInt64> multi_hist;

    auto gpair = GenerateGradientsFixedPoint(&ctx, n_samples).gpair;
    std::shared_ptr<common::HistogramCuts> cuts;

    std::size_t row_stride = 0;
    {
      /**
       * Multi page.
       */
      std::int32_t k{0};
      for (auto const& page : p_fmat->GetBatches<EllpackPage>(&ctx, p)) {
        auto impl = page.Impl();
        row_stride = impl->info.row_stride;
        if (k == 0) {
          // Initialization
          fg = std::make_unique<FeatureGroups>(impl->Cuts());
          auto init = GradientPairInt64{0, 0};
          multi_hist = decltype(multi_hist)(impl->Cuts().TotalBins(), init);
          single_hist = decltype(single_hist)(impl->Cuts().TotalBins(), init);
          cuts = std::make_shared<common::HistogramCuts>(impl->Cuts());
        }

        partitioners.emplace_back(std::make_unique<RowPartitioner>());
        partitioners.back()->Reset(&ctx, impl->Size(), impl->base_rowid);

        auto ridx = partitioners.at(k)->GetRows(0);
        auto d_histogram = dh::ToSpan(multi_hist);
        DeviceHistogramBuilder builder;
        builder.Reset(&ctx, HistMakerTrainParam::CudaDefaultNodes(), d_histogram.size(),
                      force_global);
        impl->Visit(&ctx, {}, [&](auto&& acc) {
          builder.BuildHistogram(&ctx, acc, *fg, gpair.View(ctx.Device()).Values(), ridx,
                                 d_histogram);
        });
        ++k;
      }
      ASSERT_EQ(k, n_batches);
    }

    {
      /**
       * Single page.
       */
      RowPartitioner partitioner;
      partitioner.Reset(&ctx, p_fmat->Info().num_row_, 0);

      auto concat = EllpackPageImpl(&ctx, cuts, sparsity == 0.0, row_stride, n_samples);
      std::vector<float> hess(p_fmat->Info().num_row_, 1.0f);
      std::size_t offset = 0;
      for (auto const& page : p_fmat->GetBatches<EllpackPage>(&ctx, p)) {
        bst_idx_t num_elements = concat.Copy(&ctx, page.Impl(), offset);
        offset += num_elements;
      }
      auto ridx = partitioner.GetRows(0);
      auto d_histogram = dh::ToSpan(single_hist);
      DeviceHistogramBuilder builder;
      builder.Reset(&ctx, HistMakerTrainParam::CudaDefaultNodes(), d_histogram.size(),
                    force_global);
      concat.Visit(&ctx, {}, [&](auto&& acc) {
        builder.BuildHistogram(&ctx, acc, *fg, gpair.View(ctx.Device()).Values(), ridx,
                               d_histogram);
      });
    }

    std::vector<GradientPairInt64> h_single(single_hist.size());
    thrust::copy(single_hist.begin(), single_hist.end(), h_single.begin());
    std::vector<GradientPairInt64> h_multi(multi_hist.size());
    thrust::copy(multi_hist.begin(), multi_hist.end(), h_multi.begin());

    for (std::size_t i = 0; i < single_hist.size(); ++i) {
      ASSERT_EQ(h_single[i].GetQuantisedGrad(), h_multi[i].GetQuantisedGrad()) << i;
      ASSERT_EQ(h_single[i].GetQuantisedHess(), h_multi[i].GetQuantisedHess());
    }
  }
};
}  // namespace

TEST_P(HistogramExternalMemoryTest, ExternalMemory) {
  std::apply(&HistogramExternalMemoryTest::Run, std::tuple_cat(std::make_tuple(this), GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    Histogram, HistogramExternalMemoryTest,
    ::testing::Combine(::testing::Values(0.0f, 0.2f, 0.8f), ::testing::Bool(),
                       ::testing::Values(kNoCache, kDirect, kCopy)),
    [](::testing::TestParamInfo<HistogramExternalMemoryTest::ParamType> const& info) {
      std::stringstream ss;
      auto const& p = info.param;
      ss << "sparsity_0" << (std::get<0>(p) * 10) << "_global_" << std::get<1>(p) << "_dcache_";
      switch (std::get<2>(p)) {
        case kNoCache:
          ss << "nocache";
          break;
        case kDirect:
          ss << "direct";
          break;
        case kCopy:
          ss << "copy";
          break;
      }
      return ss.str();
    });

namespace {
enum class Layout : std::int32_t { kDense = 0, kDenseMissing = 1, kSparse = 2 };

std::ostream& operator<<(std::ostream& os, Layout layout) {
  constexpr char const* kNames[] = {"dense", "missing", "sparse"};
  return os << kNames[static_cast<std::int32_t>(layout)];
}

/**
 * @brief Input with known bins. The expected histogram is computed from the input alone,
 *        it doesn't depend on any other histogram implementation.
 */
struct HistInput {
  bst_idx_t n_samples;
  bst_feature_t n_features;
  bst_bin_t n_bins;  // per feature, the largest when the counts are uneven
  // Bins of each feature. Uniform unless `skewed`, which makes `FeatureGroups` produce groups
  // of very different widths, since it packs features by bin count.
  std::vector<bst_bin_t> feature_bins;
  // Exclusive scan of `feature_bins`.
  std::vector<bst_bin_t> bin_ptrs;
  bst_target_t n_targets;
  // Bin index local to the feature, -1 for missing values. Row-major.
  std::vector<bst_bin_t> bins;
  linalg::Matrix<GradientPairInt64> gpair;
  std::vector<cuda_impl::RowIndexT> ridx;
  // Number of rows in each node.
  std::vector<std::size_t> sizes;

  HistInput(bst_idx_t n_samples, bst_feature_t n_features, bst_bin_t n_bins, bst_target_t n_targets,
            Layout layout, bool root, bool skewed = false)
      : n_samples{n_samples},
        n_features{n_features},
        n_bins{n_bins},
        feature_bins(n_features, n_bins),
        bin_ptrs(n_features + 1, 0),
        n_targets{n_targets},
        bins(n_samples * n_features),
        gpair{{n_samples, static_cast<bst_idx_t>(n_targets)}, DeviceOrd::CPU(), linalg::kF},
        ridx(n_samples) {
    if (skewed) {
      // The first half of the features have many bins, the second half have few. A group fits
      // a fixed number of bins, so the groups over the second half hold many more features.
      // The halves must not be interleaved, or every group would mix the two and the widths
      // would even out.
      for (bst_feature_t f = n_features / 2; f < n_features; ++f) {
        this->feature_bins[f] = std::max(bst_bin_t{2}, n_bins / 32);
      }
    }
    for (bst_feature_t f = 0; f < n_features; ++f) {
      this->bin_ptrs[f + 1] = this->bin_ptrs[f] + this->feature_bins[f];
    }

    std::mt19937 rng{2026};
    std::bernoulli_distribution missing_dist{0.3};
    for (bst_idx_t r = 0; r < n_samples; ++r) {
      for (bst_feature_t f = 0; f < n_features; ++f) {
        std::uniform_int_distribution<bst_bin_t> bin_dist{0, this->feature_bins[f] - 1};
        bool missing = false;
        switch (layout) {
          case Layout::kDense:
            break;
          case Layout::kDenseMissing:
            // A complete row keeps the row stride equal to the number of features.
            missing = r != 0 && missing_dist(rng);
            break;
          case Layout::kSparse:
            // No row is complete.
            missing = f == r % n_features || missing_dist(rng);
            break;
        }
        this->bins[r * n_features + f] = missing ? -1 : bin_dist(rng);
      }
    }

    std::uniform_int_distribution<std::int64_t> grad_dist{-(1 << 20), 1 << 20};
    std::uniform_int_distribution<std::int64_t> hess_dist{0, 1 << 20};
    for (auto& v : this->gpair.Data()->HostVector()) {
      v = GradientPairInt64{grad_dist(rng), hess_dist(rng)};
    }

    std::iota(this->ridx.begin(), this->ridx.end(), 0);
    if (root) {
      this->sizes = {n_samples};
    } else {
      std::shuffle(this->ridx.begin(), this->ridx.end(), rng);
      // Empty nodes and nodes smaller than a tile.
      this->sizes = {0, 1, 7, 0, 1000};
      auto n_used = std::accumulate(this->sizes.cbegin(), this->sizes.cend(), std::size_t{0});
      CHECK_GE(n_samples, n_used);
      this->sizes.push_back(n_samples - n_used);
      this->sizes.push_back(0);
    }
  }

  // Cut values are `b + 1` for bin `b` and the feature values are `b + 0.5`.
  [[nodiscard]] std::unique_ptr<EllpackPageImpl> MakeEllpack(Context const* ctx) const {
    auto p_cuts = std::make_shared<common::HistogramCuts>(this->n_features);
    std::vector<std::uint32_t> ptrs(this->n_features + 1);
    std::vector<float> cut_values;
    for (bst_feature_t f = 0; f < this->n_features; ++f) {
      ptrs[f + 1] = this->bin_ptrs[f + 1];
      for (bst_bin_t b = 0; b < this->feature_bins[f]; ++b) {
        cut_values.push_back(b + 1.0f);
      }
    }
    p_cuts->cut_ptrs_.HostVector() = std::move(ptrs);
    p_cuts->cut_values_.HostVector() = std::move(cut_values);

    auto missing = std::numeric_limits<float>::quiet_NaN();
    linalg::Matrix<float> x{{this->n_samples, static_cast<bst_idx_t>(this->n_features)},
                            DeviceOrd::CPU()};
    auto& h_x = x.Data()->HostVector();
    std::transform(this->bins.cbegin(), this->bins.cend(), h_x.begin(),
                   [&](bst_bin_t b) { return b < 0 ? missing : b + 0.5f; });

    auto str = linalg::ArrayInterfaceStr(x.View(ctx->Device()));
    auto adapter = data::CupyAdapter{StringView{str}};
    dh::device_vector<bst_idx_t> row_counts(this->n_samples);
    auto row_stride =
        data::GetRowCounts(ctx, adapter.Value(), dh::ToSpan(row_counts), ctx->Device(), missing);
    bool is_dense =
        std::none_of(this->bins.cbegin(), this->bins.cend(), [](bst_bin_t b) { return b < 0; });
    return std::make_unique<EllpackPageImpl>(
        ctx, adapter.Value(), missing, is_dense, dh::ToSpan(row_counts),
        common::Span<FeatureType const>{}, row_stride, this->n_samples, p_cuts);
  }

  // One target-major histogram for each node.
  [[nodiscard]] std::vector<std::vector<GradientPairInt64>> Expected() {
    auto n_total_bins = this->bin_ptrs.back();
    auto h_gpair = this->gpair.HostView();
    std::vector<std::vector<GradientPairInt64>> hists;
    std::size_t beg = 0;
    for (auto n_rows : this->sizes) {
      auto& hist = hists.emplace_back(n_total_bins * this->n_targets);
      for (std::size_t i = beg; i < beg + n_rows; ++i) {
        auto r = this->ridx[i];
        for (bst_feature_t f = 0; f < this->n_features; ++f) {
          auto b = this->bins[r * this->n_features + f];
          if (b < 0) {
            continue;
          }
          for (bst_target_t t = 0; t < this->n_targets; ++t) {
            hist[t * n_total_bins + this->bin_ptrs[f] + b] += h_gpair(r, t);
          }
        }
      }
      beg += n_rows;
    }
    return hists;
  }
};

// Properties of the built input, for tests that require them.
struct BuildInfo {
  bst_idx_t n_symbols{0};
  std::size_t n_groups{0};
  // The narrowest and widest feature group.
  bst_feature_t min_group_features{0};
  bst_feature_t max_group_features{0};
};

void TestBuildHistogram(bst_idx_t n_samples, bst_feature_t n_features, bst_bin_t n_bins,
                        bst_target_t n_targets, Layout layout, bool root, bool force_global,
                        bool small_groups, BuildInfo* info = nullptr, bool skewed = false) {
  auto ctx = MakeCUDACtx(0);
  HistInput input{n_samples, n_features, n_bins, n_targets, layout, root, skewed};
  auto expected = input.Expected();

  auto page = input.MakeEllpack(&ctx);
  ASSERT_EQ(page->IsDense(), layout == Layout::kDense);
  ASSERT_EQ(page->IsDenseCompressed(), layout != Layout::kSparse);

  auto shmem_bytes =
      n_targets == 1 ? DftStHistShmemBytes(ctx.Ordinal()) : DftMtHistShmemBytes(ctx.Ordinal());
  if (small_groups) {
    // Four features in each group, the nodes are split into many segments.
    shmem_bytes = sizeof(GradientPairInt64) * n_bins * 4;
  }
  FeatureGroups fg{page->Cuts(), page->IsDenseCompressed(), shmem_bytes};
  if (small_groups && page->IsDenseCompressed()) {
    ASSERT_GT(fg.feature_segments.Size(), 3);
  }
  if (info) {
    auto const& h_fs = fg.feature_segments.ConstHostVector();
    bst_feature_t min_w = std::numeric_limits<bst_feature_t>::max(), max_w = 0;
    for (std::size_t i = 1; i < h_fs.size(); ++i) {
      min_w = std::min(min_w, h_fs[i] - h_fs[i - 1]);
      max_w = std::max(max_w, h_fs[i] - h_fs[i - 1]);
    }
    *info = BuildInfo{page->NumSymbols(), h_fs.size() - 1, min_w, max_w};
  }

  bst_node_t n_nodes = input.sizes.size();
  DeviceHistogramBuilder builder;
  builder.Reset(&ctx, n_nodes, page->Cuts().TotalBins() * n_targets, force_global);
  std::vector<bst_node_t> nidx(n_nodes);
  std::iota(nidx.begin(), nidx.end(), 0);
  builder.AllocateHistograms(&ctx, nidx);

  dh::device_vector<cuda_impl::RowIndexT> ridx{input.ridx};
  std::vector<common::Span<cuda_impl::RowIndexT const>> ridxs;
  std::vector<common::Span<GradientPairInt64>> hists;
  std::size_t beg = 0;
  for (bst_node_t i = 0; i < n_nodes; ++i) {
    ridxs.emplace_back(dh::ToSpan(ridx).subspan(beg, input.sizes[i]));
    hists.push_back(builder.GetNodeHistogram(i));
    beg += input.sizes[i];
  }
  builder.BuildHistogram(&ctx, page->GetDeviceEllpack(&ctx, {}), fg, input.gpair.View(ctx.Device()),
                         ridxs, hists);

  for (bst_node_t i = 0; i < n_nodes; ++i) {
    std::vector<GradientPairInt64> got(hists[i].size());
    dh::CopyDeviceSpanToVector(&got, hists[i]);
    ASSERT_EQ(got.size(), expected[i].size());
    for (std::size_t j = 0; j < got.size(); ++j) {
      ASSERT_EQ(got[j], expected[i][j]) << "node:" << i << " bin:" << j;
    }
  }
}

class HistogramBuildTest
    : public ::testing::TestWithParam<std::tuple<Layout, bst_target_t, bool, bool, bool>> {};

std::string HistogramBuildName(
    ::testing::TestParamInfo<HistogramBuildTest::ParamType> const& info) {
  auto [layout, n_targets, root, force_global, small_groups] = info.param;
  std::stringstream ss;
  ss << layout << "_targets_" << n_targets << (root ? "_root" : "_nodes")
     << (force_global ? "_global" : "_shared") << (small_groups ? "_small_groups" : "_dft_groups");
  return ss.str();
}
}  // namespace

TEST_P(HistogramBuildTest, Build) {
  auto [layout, n_targets, root, force_global, small_groups] = this->GetParam();
  TestBuildHistogram(1 << 13, 37, 16, n_targets, layout, root, force_global, small_groups);
}

INSTANTIATE_TEST_SUITE_P(
    Histogram, HistogramBuildTest,
    ::testing::Combine(::testing::Values(Layout::kDense, Layout::kDenseMissing, Layout::kSparse),
                       ::testing::Values<bst_target_t>(1, 3), ::testing::Bool(), ::testing::Bool(),
                       ::testing::Bool()),
    HistogramBuildName);

// Multiple tiles for each block. Blocks take multiple tiles only when the tiles outnumber
// the resident blocks, the rows scale with the number of SMs to keep at least four tiles
// for each SM.
TEST(Histogram, BuildLarge) {
  auto n_samples = std::max<bst_idx_t>(1 << 21, static_cast<bst_idx_t>(curt::GetMpCnt(0)) << 14);
  for (bst_target_t n_targets : {1, 2}) {
    for (auto force_global : {false, true}) {
      TestBuildHistogram(n_samples, 2, 256, n_targets, Layout::kDense, /*root=*/false, force_global,
                         /*small_groups=*/false);
    }
  }
}

// Many bins. The sparse page stores global bin indices wider than 16 bits, and the dense
// pages need multiple feature groups with the default shared memory budget.
TEST(Histogram, BuildWide) {
  for (auto layout : {Layout::kDense, Layout::kDenseMissing, Layout::kSparse}) {
    for (bst_target_t n_targets : {1, 3}) {
      for (auto force_global : {false, true}) {
        BuildInfo info;
        ASSERT_NO_FATAL_FAILURE(TestBuildHistogram(1 << 12, 257, 256, n_targets, layout,
                                                   /*root=*/false, force_global,
                                                   /*small_groups=*/false, &info));
        if (layout == Layout::kSparse) {
          ASSERT_GT(info.n_symbols, 1 << 16);
        } else {
          ASSERT_GT(info.n_groups, 1);
        }
      }
    }
  }
}

// `FeatureGroups` packs features until the bin budget is full, so uneven bin counts give
// groups of very different widths. The grid covers the widest, and a block's chunk is a number
// of entries, so the work of a block is the same in every group.
TEST(Histogram, BuildSkewedGroups) {
  for (auto layout : {Layout::kDense, Layout::kDenseMissing, Layout::kSparse}) {
    for (bst_target_t n_targets : {1, 3}) {
      for (auto force_global : {false, true}) {
        BuildInfo info;
        ASSERT_NO_FATAL_FAILURE(TestBuildHistogram(1 << 12, 192, 256, n_targets, layout,
                                                   /*root=*/false, force_global,
                                                   /*small_groups=*/false, &info,
                                                   /*skewed=*/true));
        if (layout != Layout::kSparse) {
          ASSERT_GT(info.n_groups, 1);
          // The point of the test: the groups are not all the same width.
          ASSERT_GT(info.max_group_features, info.min_group_features * 2);
        }
      }
    }
  }
}
}  // namespace xgboost::tree
