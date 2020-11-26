/*!
 * Copyright 2015-2022 XGBoost contributors
 */
#include <xgboost/logging.h>
#include <xgboost/objective.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../common/device_helpers.cuh"
#include "../common/math.h"
#include "../common/random.h"
#include "../common/ranking_utils.cuh"
#include "../common/ranking_utils.h"
#include "rank_obj.cuh"
#include "rank_obj.h"
#include "xgboost/json.h"
#include "xgboost/parameter.h"

namespace xgboost {
namespace obj {
DMLC_REGISTRY_FILE_TAG(rank_obj_gpu);

struct DeviceNDCGCache {
  MetaInfo const *p_info{nullptr};
  dh::device_vector<float> inv_IDCG;  // NOLINT
  size_t truncation{0};

  dh::device_vector<size_t> sorted_idx_cache;  // avoid allocating memory for each iter.

  dh::device_vector<size_t> threads_group_ptr;
  dh::device_vector<bst_group_t> group_ptr;
  size_t n_threads{0};

  explicit DeviceNDCGCache(MetaInfo const *info, size_t ndcg_truncation,
                           common::Span<float const> predts)
      : p_info{info}, truncation{ndcg_truncation} {
    auto labels = info->labels.View(dh::CurrentDevice());
    auto const &h_group_ptr = info->group_ptr_;
    group_ptr.resize(h_group_ptr.size());
    thrust::copy(h_group_ptr.cbegin(), h_group_ptr.cend(), group_ptr.begin());
    auto d_group_ptr = dh::ToSpan(group_ptr);

    size_t n_groups = group_ptr.size() - 1;
    inv_IDCG.resize(n_groups, 0.0f);
    CalcQueriesInvIDCG(labels.Values(), d_group_ptr, dh::ToSpan(inv_IDCG), ndcg_truncation);
    CHECK_GE(ndcg_truncation, 1ul);

    threads_group_ptr.resize(n_groups + 1, 0);
    auto d_threads_group_ptr = dh::ToSpan(threads_group_ptr);
    n_threads = ::xgboost::common::SegmentedTrapezoidThreads(d_group_ptr, d_threads_group_ptr,
                                                             ndcg_truncation);

    sorted_idx_cache.resize(labels.Size(), 0);
  }

  size_t Groups() const { return group_ptr.size() - 1; }
  size_t Threads() const { return n_threads; }
  common::Span<size_t const> ThreadsGroupPtr() const {
    return {threads_group_ptr.data().get(), threads_group_ptr.size()};
  }
  common::Span<bst_group_t const> DataGroupPtr() const {
    return {group_ptr.data().get(), group_ptr.size()};
  }
  common::Span<float const> InvIDCG() const { return {inv_IDCG.data().get(), inv_IDCG.size()}; }

  auto SortedIdx(common::Span<float const> predts) {
    auto d_sorted_idx = dh::ToSpan(sorted_idx_cache);
    auto d_group_ptr = DataGroupPtr();
    dh::SegmentedSequence(d_group_ptr, d_sorted_idx);
    dh::SegmentedArgSort<true>(predts, d_group_ptr, d_sorted_idx);
    return d_sorted_idx;
  }
};

void LambdaMARTGetGradientNDCGGPUKernel(const HostDeviceVector<bst_float> &preds,
                                        const MetaInfo &info, size_t ndcg_truncation,
                                        std::shared_ptr<DeviceNDCGCache> *cache, int32_t device_id,
                                        HostDeviceVector<GradientPair> *out_gpair) {
  dh::safe_cuda(cudaSetDevice(device_id));
  size_t n_groups = info.group_ptr_.size() - 1;

  info.labels.SetDevice(device_id);
  preds.SetDevice(device_id);
  out_gpair->SetDevice(device_id);
  out_gpair->Resize(preds.Size());

  auto labels = info.labels.View(dh::CurrentDevice());
  auto predts = preds.ConstDeviceSpan();
  auto gpairs = out_gpair->DeviceSpan();
  dh::LaunchN(gpairs.size(), [=] XGBOOST_DEVICE(size_t idx) { gpairs[idx] = GradientPair{}; });

  auto &p_cache = *cache;
  if (!p_cache || p_cache->p_info != &info || p_cache->truncation != ndcg_truncation) {
    p_cache = std::make_shared<DeviceNDCGCache>(&info, ndcg_truncation, predts);
  }
  auto d_threads_group_ptr = p_cache->ThreadsGroupPtr();
  auto d_group_ptr = p_cache->DataGroupPtr();
  auto d_inv_IDCG = p_cache->InvIDCG();
  auto d_sorted_idx = p_cache->SortedIdx(predts);

  dh::LaunchN(p_cache->Threads(), [=] XGBOOST_DEVICE(size_t idx) {
    auto query_group_idx = dh::SegmentId(d_threads_group_ptr, idx);

    auto thread_group_begin = d_threads_group_ptr[query_group_idx];
    auto n_group_threads = d_threads_group_ptr[query_group_idx + 1] - thread_group_begin;
    auto idx_in_thread_group = idx - thread_group_begin;

    auto data_group_begin = d_group_ptr[query_group_idx];
    size_t n_data = d_group_ptr[query_group_idx + 1] - data_group_begin;

    auto inv_IDCG = d_inv_IDCG[query_group_idx];

    auto g_labels = labels.Slice(
        linalg::Range(static_cast<size_t>(data_group_begin), data_group_begin + n_data));
    auto g_predts = predts.subspan(data_group_begin, n_data);
    auto g_gpairs = gpairs.subspan(data_group_begin, n_data);
    auto g_sorted_idx = d_sorted_idx.subspan(data_group_begin, n_data);

    size_t i = 0, j = 0;
    common::UnravelTrapeziodIdx(idx_in_thread_group, n_data, &i, &j);
    LambdaNDCG(g_labels.Values(), g_predts, g_sorted_idx, i, j, inv_IDCG, g_gpairs);
  });

  if (!info.weights_.Empty()) {
    info.weights_.SetDevice(device_id);
    auto d_weights = info.weights_.ConstDeviceSpan();
    CHECK_EQ(d_weights.size(), n_groups);
    dh::LaunchN(info.num_row_, [=] XGBOOST_DEVICE(size_t idx) {
      auto g = dh::SegmentId(d_group_ptr, idx);
      gpairs[idx] *= d_weights[g];
    });
  }
}

void CheckNDCGLabelsGPUKernel(LambdaMARTParam const &p, common::Span<float const> labels) {
  auto label_is_integer =
      thrust::none_of(labels.data(), labels.data() + labels.size(), [](auto const &v) {
        auto l = std::floor(v);
        return std::fabs(l - v) > kRtEps || v < 0.0f;
      });
  CHECK(label_is_integer) << "When using relevance degree as target, labels "
                             "must be either 0 or positive integer.";
}
}  // namespace obj
}  // namespace xgboost
