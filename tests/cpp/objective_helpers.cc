/**
 * Copyright (c) 2023, XGBoost contributors
 */
#include "objective_helpers.h"

#include "../../src/common/linalg_op.h"  // for begin, end
#include "helpers.h"                     // for RandomDataGenerator

namespace xgboost {
std::shared_ptr<DMatrix> MakeFmatForObjTest(std::string const& obj) {
  auto constexpr kRows = 10, kCols = 10;
  auto p_fmat = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true);
  auto& h_upper = p_fmat->Info().labels_upper_bound_.HostVector();
  auto& h_lower = p_fmat->Info().labels_lower_bound_.HostVector();
  h_lower.resize(kRows);
  h_upper.resize(kRows);
  for (size_t i = 0; i < kRows; ++i) {
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
  return p_fmat;
};
}  // namespace xgboost
