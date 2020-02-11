/*!
 * Copyright 2015-2020 by Contributors
 * \file tree_updater.cc
 * \brief Registry of tree updaters.
 */
#include <dmlc/registry.h>

#include "xgboost/tree_updater.h"
#include "xgboost/host_device_vector.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::TreeUpdaterReg);
}  // namespace dmlc

namespace xgboost {

TreeUpdater* TreeUpdater::Create(const std::string& name, GenericParameter const* tparam) {
  auto *e = ::dmlc::Registry< ::xgboost::TreeUpdaterReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown tree updater " << name;
  }
  auto p_updater = (e->body)();
  p_updater->tparam_ = tparam;
  return p_updater;
}


void TreeUpdater::LeafIndexContainer::Init(size_t num_rows, int device) {
  this->Clear();

  Segment root {0, num_rows};

  ridx_segments_.emplace_back(root);
  if (device >= 0) {
    row_index_.SetDevice(device);
    row_index_.DeviceSpan();
  }
  row_index_.Resize(root.end);
}

common::Span<size_t const> TreeUpdater::LeafIndexContainer::HostRows(int32_t node_id) const {
  return const_cast<LeafIndexContainer*>(this)->HostRows(node_id);
}

common::Span<size_t> TreeUpdater::LeafIndexContainer::HostRows(int32_t node_id) {
  Segment seg { ridx_segments_.at(node_id) };
  CHECK_LE(seg.end, this->row_index_.Size()) << "node_id: " << node_id;
  // Avoids calling __host__ inside __host__ __device__
  size_t* ptr = row_index_.HostPointer();
  auto size = row_index_.Size();
  if (seg.begin == seg.end) {
    return common::Span<size_t>{};
  } else {
    return common::Span<size_t>{ptr, size}.subspan(
        seg.begin, seg.end - seg.begin);
  }
}

common::Span<size_t const> TreeUpdater::LeafIndexContainer::DeviceRows(int32_t node_id) const {
  return const_cast<LeafIndexContainer*>(this)->DeviceRows(node_id);
}

common::Span<size_t> TreeUpdater::LeafIndexContainer::DeviceRows(int32_t node_id) {
  Segment segment { ridx_segments_.at(node_id) };
  CHECK_LE(segment.end, this->row_index_.Size());
  auto span = row_index_.DeviceSpan();
  if (segment.Size() == 0) {
    return common::Span<size_t>();
  }
  return span.subspan(segment.begin, segment.end - segment.begin);
}

void TreeUpdater::LeafIndexContainer::AddSplit(int32_t node_id, size_t left_count,
                                            int32_t left_node_id, int32_t right_node_id) {
  Segment e = ridx_segments_.at(node_id);
  CHECK_NE(e.end, 0) << "node_id:" << node_id << ", left:" << left_count;

  size_t begin = e.begin;
  size_t left_end = begin + left_count;

  CHECK_GT(ridx_segments_.size(), 0);
  ridx_segments_.resize(std::max(static_cast<int32_t>(ridx_segments_.size()),
                                std::max(left_node_id, right_node_id) + 1), Segment(0, 0));

  // ridx_segments_.
  CHECK_LE(left_end, this->row_index_.Size());
  ridx_segments_.at(left_node_id) = Segment(begin, left_end);
  CHECK_LE(e.end, this->row_index_.Size()) << node_id;
  ridx_segments_.at(right_node_id) = Segment(left_end, e.end);
}

}  // namespace xgboost

namespace xgboost {
namespace tree {
// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(updater_colmaker);
DMLC_REGISTRY_LINK_TAG(updater_skmaker);
DMLC_REGISTRY_LINK_TAG(updater_refresh);
DMLC_REGISTRY_LINK_TAG(updater_prune);
DMLC_REGISTRY_LINK_TAG(updater_quantile_hist);
DMLC_REGISTRY_LINK_TAG(updater_histmaker);
DMLC_REGISTRY_LINK_TAG(updater_sync);
#ifdef XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(updater_gpu_hist);
#endif  // XGBOOST_USE_CUDA
}  // namespace tree
}  // namespace xgboost
