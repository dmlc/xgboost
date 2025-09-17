/*!
 * Copyright 2017-2025 by Contributors
 * \file hist_dispatcher.h
 */
#ifndef PLUGIN_SYCL_TREE_HIST_DISPATCHER_H_
#define PLUGIN_SYCL_TREE_HIST_DISPATCHER_H_

#include <algorithm>
#include <sycl/sycl.hpp>

#include "../device_properties.h"

namespace xgboost {
namespace sycl {
namespace tree {

struct BlockParams { size_t size, nblocks; };

template <typename FPType>
class HistDispatcher {
 public:
  // Max n_blocks/max_compute_units ration.
  // Higher -> better GPU utilisation with higer memory overhead.
  constexpr static int kMaxGPUUtilisation = 4;
  // Minimal value of block size for buffer-based hist building
  constexpr static size_t KMinBlockSize = 32;
  // Maximal value of block size, when increasing can affect performance
  constexpr static size_t KMaxEffectiveBlockSize = 1u << 11;
  // Maximal number of bins acceptable for local histograms
  constexpr static size_t KMaxNumBins = 256;
  // Amount of sram for local-histogram kernel launch
  constexpr static float KLocalHistSRAM = 32. * 1024;
  // Max workgroups size, used by atomic-based hist-building
  constexpr static size_t kMaxWorkGroupSizeAtomic = 32;
  // Max workgroups size, used for local histograms
  constexpr static size_t kMaxWorkGroupSizeLocal = 256;
  // Atomic efficency normalization
  constexpr static float kAtomicEfficiencyNormalization = 16 * 1024;
  // Block kernel launch penalty normalization
  constexpr static float kBlockPenaltyNormalization = 32 * 1024;
  // Relative weight of quadratic term in atomic penalty model
  constexpr static float kAtomicQuadraticWeight = 1.0 / 8.0;
  // Minimal value of threshold GPU load
  constexpr static float kMinTh = 1.0 / 16.0;

  bool use_local_hist = false;
  bool use_atomics = false;
  size_t work_group_size;
  BlockParams block;

  inline BlockParams GetBlocksParameters(size_t size, size_t max_nblocks,
                                         size_t max_compute_units) const {
    if (max_nblocks == 0) return {0, 0};
    size_t nblocks = max_compute_units;

    size_t block_size = size / nblocks + !!(size % nblocks);
    while (block_size > (1u << 11)) {
      nblocks *= 2;
      if (nblocks >= max_nblocks) {
        nblocks = max_nblocks;
        block_size = size / nblocks + !!(size % nblocks);
        break;
      }
      block_size = size / nblocks + !!(size % nblocks);
    }

    if (block_size < KMinBlockSize) {
      block_size = KMinBlockSize;
      nblocks = size / block_size + !!(size % block_size);
    }

    return {block_size, nblocks};
  }

  HistDispatcher(const DeviceProperties& device_prop, bool isDense, size_t size,
                 size_t max_nblocks, size_t nbins, size_t ncolumns,
                 size_t max_num_bins, size_t min_num_bins) {
    block = GetBlocksParameters(size, max_nblocks, device_prop.max_compute_units);
    work_group_size = std::min(ncolumns, device_prop.max_work_group_size);
    if (!device_prop.is_gpu) return;

    using GradientPairT = xgboost::detail::GradientPairInternal<FPType>;
    /* If local histogram is possible and beneficial */
    const int buff_size = nbins * sizeof(GradientPairT);
    /* block_size writes into array of size max_num_bins are made,
    * if (block_size < max_num_bins)
    * most part of buffer isn't used and perf suffers.
    */
    const size_t th_block_size = max_num_bins;
    use_local_hist = (buff_size < device_prop.sram_size_per_eu - KLocalHistSRAM)
                      && isDense
                      && (max_num_bins <= KMaxNumBins)
                      && (block.size >= th_block_size);

    /* Predict penalty from atomic usage and compare with one from block-based build with buffer */
    // EUs processing different columns do not trigger conflicts.
    float wg_per_columns = std::max(1.0f, static_cast<float>(ncolumns) / kMaxWorkGroupSizeAtomic);
    /* Rows are processed per execution unit.
    * Some EUs process different columns, and don't triiger conflicts.
    * We use a worse case scenario, i.e. use the minimal number of bins per feature
    */
    float conflicts_per_bin = (device_prop.max_compute_units / wg_per_columns) / min_num_bins;

    // Atomics resolve conflicts between EUs, so L2 size can be a proxy for atomic efficiency.
    float atomic_efficency = device_prop.l2_size_per_eu / kAtomicEfficiencyNormalization;
    // We use simple quadratic model to predict atomic penalty
    float atomic_penalty = conflicts_per_bin
                        + kAtomicQuadraticWeight * (conflicts_per_bin * conflicts_per_bin);

    // Block-based builder operates with buffer of type FPType, placed in L2.
    float base_block_penalty = kBlockPenaltyNormalization /
                                device_prop.l2_size_per_eu * (sizeof(FPType) / 4);

    if (block.nblocks >= device_prop.max_compute_units) {
      // if GPU is fully loaded, we can simply compare penaltys.
      use_atomics = base_block_penalty > atomic_penalty / atomic_efficency;
    } else {
      float blocks_per_eu = static_cast<float>(block.nblocks) / device_prop.max_compute_units;
      /* The GPU is not 100% loaded. We need to take this into account in our model:
      * block_penalty = base_block_penalty + base_time * (1 - blocks_per_eu);
      *
      * atomics should be used, if:
      * block_penalty > atomic_penalty
      *
      * The normalization is chosen so that: base_time = 1
      * base_block_penalty + 1 - blocks_per_eu > atomic_penalty / atomic_efficency
      *
      * blocks_per_eu < 1 + base_block_penalty - atomic_penalty / atomic_efficency
      */
      float th_block_per_eu = 1 + base_block_penalty - atomic_penalty / atomic_efficency;

      /* We can't trust the decision of the approximate performance model
      * if penalties are close to each other
      * i.e. (1 + base_block_penalty) ~ (atomic_penalty / atomic_efficency)
      * We manually limit the minimal value of th_block_per_eu,
      * to determine the behaviour in this region.
      */
      th_block_per_eu = std::max<float>(kMinTh, th_block_per_eu);

      use_atomics = (blocks_per_eu < th_block_per_eu);
    }

    if (use_atomics) {
      work_group_size = std::min(kMaxWorkGroupSizeAtomic,
                                 work_group_size);
    } else if (use_local_hist) {
      work_group_size = std::min(kMaxWorkGroupSizeLocal,
                                 work_group_size);
    }
  }
};

// For some datasets buffer is not used, we estimate if it is the case.
template<typename FPType>
size_t GetRequiredBufferSize(const DeviceProperties& device_prop, size_t max_n_rows, size_t nbins,
                             size_t ncolumns, size_t max_num_bins, size_t min_num_bins) {
  size_t max_nblocks = HistDispatcher<FPType>::kMaxGPUUtilisation * device_prop.max_compute_units;
  // Buffer size doesn't depend on isDense flag.
  auto build_params = HistDispatcher<FPType>
                      (device_prop, true, max_n_rows, max_nblocks, nbins,
                       ncolumns, max_num_bins, min_num_bins);

  return build_params.use_atomics ? 0 : build_params.block.nblocks;
}

}  // namespace tree
}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_TREE_HIST_DISPATCHER_H_
