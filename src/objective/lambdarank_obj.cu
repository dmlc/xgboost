/**
 * Copyright 2015-2023 by XGBoost contributors
 *
 * \brief CUDA implementation of lambdarank.
 */
#include <thrust/fill.h>                        // for fill_n
#include <thrust/for_each.h>                    // for for_each_n
#include <thrust/iterator/counting_iterator.h>  // for make_counting_iterator
#include <thrust/iterator/zip_iterator.h>       // for make_zip_iterator
#include <thrust/tuple.h>                       // for make_tuple, tuple, tie, get

#include <algorithm>                            // for min
#include <cassert>                              // for assert
#include <cmath>                                // for abs, log2, isinf
#include <cstddef>                              // for size_t
#include <cstdint>                              // for int32_t
#include <memory>                               // for shared_ptr
#include <utility>

#include "../common/algorithm.cuh"       // for SegmentedArgSort
#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/deterministic.cuh"   // for CreateRoundingFactor, TruncateWithRounding
#include "../common/device_helpers.cuh"  // for SegmentId, TemporaryArray, AtomicAddGpair
#include "../common/optional_weight.h"   // for MakeOptionalWeights
#include "../common/ranking_utils.h"     // for NDCGCache, LambdaRankParam, rel_degree_t
#include "lambdarank_obj.cuh"
#include "lambdarank_obj.h"
#include "xgboost/base.h"                // for bst_group_t, XGBOOST_DEVICE, GradientPair
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for VectorView, Range, Vector
#include "xgboost/logging.h"
#include "xgboost/span.h"                // for Span

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(lambdarank_obj_cu);

namespace cuda_impl {
common::Span<std::size_t const> SortY(Context const* ctx, MetaInfo const& info,
                                      common::Span<std::size_t const> d_rank,
                                      std::shared_ptr<ltr::RankingCache> p_cache) {
  auto const d_group_ptr = p_cache->DataGroupPtr(ctx);
  auto label = info.labels.View(ctx->gpu_id);
  // The buffer for ranked y is necessary as cub segmented sort accepts only pointer.
  auto d_y_ranked = p_cache->RankedY(ctx, info.num_row_);
  thrust::for_each_n(ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(0ul), d_y_ranked.size(),
                     [=] XGBOOST_DEVICE(std::size_t i) {
                       auto g = dh::SegmentId(d_group_ptr, i);
                       auto g_label =
                           label.Slice(linalg::Range(d_group_ptr[g], d_group_ptr[g + 1]), 0);
                       auto g_rank_idx = d_rank.subspan(d_group_ptr[g], g_label.Size());
                       i -= d_group_ptr[g];
                       auto g_y_ranked = d_y_ranked.subspan(d_group_ptr[g], g_label.Size());
                       g_y_ranked[i] = g_label(g_rank_idx[i]);
                     });
  auto d_y_sorted_idx = p_cache->SortedIdxY(ctx, info.num_row_);
  common::SegmentedArgSort<false, true>(ctx, d_y_ranked, d_group_ptr, d_y_sorted_idx);
  return d_y_sorted_idx;
}
}  // namespace cuda_impl
}  // namespace xgboost::obj
