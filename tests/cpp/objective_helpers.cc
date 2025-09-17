/**
 * Copyright 2023-2025, XGBoost contributors
 */
#include "objective_helpers.h"

#include "../../src/common/linalg_op.h"  // for begin, end
#include "helpers.h"                     // for RandomDataGenerator

namespace xgboost {

void MakeLabelForObjTest(std::shared_ptr<DMatrix> p_fmat, std::string const& obj) {
  auto& h_upper = p_fmat->Info().labels_upper_bound_.HostVector();
  auto& h_lower = p_fmat->Info().labels_lower_bound_.HostVector();
  h_lower.resize(p_fmat->Info().num_row_);
  h_upper.resize(p_fmat->Info().num_row_);
  for (size_t i = 0; i < p_fmat->Info().num_row_; ++i) {
    h_lower[i] = 1;
    h_upper[i] = 10;
  }

  if (obj.find("rank:") != std::string::npos) {
    auto h_label = p_fmat->Info().labels.HostView();
    std::size_t k = 0;
    for (auto& v : h_label) {
      v = k % 2 == 0;
      ++k;
    }
  }
}

[[nodiscard]] std::shared_ptr<DMatrix> MakeFmatForObjTest(std::string const& obj,
                                                          bst_idx_t n_samples,
                                                          bst_feature_t n_features,
                                                          bst_target_t n_classes, bool make_label) {
  std::shared_ptr<DMatrix> p_fmat;
  if (obj.find("multi:") != std::string::npos) {
    CHECK_GE(n_classes, 3);
    p_fmat = RandomDataGenerator{n_samples, n_features, 0}.Classes(n_classes).GenerateDMatrix(
        make_label);
  } else {
    p_fmat = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(make_label);
  }
  if (make_label) {
    MakeLabelForObjTest(p_fmat, obj);
  }
  return p_fmat;
}
}  // namespace xgboost
