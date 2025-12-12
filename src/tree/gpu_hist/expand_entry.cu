/**
 * Copyright 2025, XGBoost Contributors
 */
#include <cstddef>  // for size_t
#include <ostream>  // for ostream
#include <vector>   // for vector

#include "../../common/device_helpers.cuh"  // for CopyDeviceSpanToVector
#include "../../common/type.h"              // for GetValueT
#include "expand_entry.cuh"

namespace xgboost::tree::cuda_impl {
std::ostream& operator<<(std::ostream& os, MultiExpandEntry const& e) {
  os << "MultiExpandEntry:\n"
     << "nidx: " << e.nidx << "\n"
     << "depth: " << e.depth << "\n"
     << "loss: " << e.split.loss_chg << "\n";

  std::vector<GradientPairInt64> h_node_sum(e.split.child_sum.size());
  dh::CopyDeviceSpanToVector(&h_node_sum, e.split.child_sum);

  auto print_span = [&](auto const& span) {
    using T = typename common::GetValueT<decltype(span)>::value_type;
    std::vector<T> h_vec(span.size());
    dh::CopyDeviceSpanToVector(&h_vec, span);

    os << "[";
    for (std::size_t i = 0; i < h_vec.size(); ++i) {
      os << h_vec[i];
      if (i != h_vec.size() - 1) {
        os << ", ";
      }
    }
    os << "]\n";
  };
  if (e.split.dir == kRightDir) {
    os << "left_sum: ";
  } else {
    os << "right_sum: ";
  }
  print_span(e.split.child_sum);

  os << "base_weight: ";
  print_span(e.base_weight);
  os << "left_weight: ";
  print_span(e.left_weight);
  os << "right_weight: ";
  print_span(e.right_weight);

  return os;
}
}  // namespace xgboost::tree::cuda_impl
