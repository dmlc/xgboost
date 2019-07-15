#ifndef XGBOOST_TEST_GBM_H_
#define XGBOOST_TEST_GBM_H_

#include <xgboost/base.h>
#include <xgboost/gbm.h>
#include <xgboost/generic_parameters.h>

#include <memory>
#include <string>

#include "../helpers.h"

namespace xgboost {

inline std::unique_ptr<GradientBooster> ConstructGBM(
    std::string name, Args kwargs, size_t kRows, size_t kCols) {
  GenericParameter param;
  param.Init(Args{});
  std::unique_ptr<GradientBooster> gbm {
    GradientBooster::Create(name, &param, {}, 0)};
  gbm->Configure(kwargs);
  auto pp_dmat = CreateDMatrix(kRows, kCols, 0);
  auto p_dmat = *pp_dmat;

  std::vector<float> labels(kRows);
  for (size_t i = 0; i < kRows; ++i) {
    labels[i] = i;
  }
  p_dmat->Info().labels_.HostVector() = labels;
  HostDeviceVector<GradientPair> gpair;
  auto& h_gpair = gpair.HostVector();
  h_gpair.resize(kRows);
  for (size_t i = 0; i < kRows; ++i) {
    h_gpair[i] = {static_cast<float>(i), 1};
  }

  gbm->DoBoost(p_dmat.get(), &gpair, nullptr);

  delete pp_dmat;
  return gbm;
}

}  // namespace xgboost

#endif  // XGBOOST_TEST_GBM_H_
