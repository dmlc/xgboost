/*!
 * Copyright 2018 XGBoost contributors
 */
#ifndef XGBOOST_TREE_UPDATER_GPU_H_
#define XGBOOST_TREE_UPDATER_GPU_H_

#include <xgboost/tree_updater.h>

#include <memory>
#include <utility>
#include <vector>
#include <string>

namespace xgboost {
namespace tree {

class GPUExactMakerImpl;

class GPUMaker : public TreeUpdater {
 protected:
  GPUExactMakerImpl *pimpl_;

 public:
  GPUMaker();
  ~GPUMaker() override;

  void Init(const std::vector<std::pair<std::string, std::string>> &args) override;
  void Update(HostDeviceVector<GradientPair> *gpair, DMatrix *dmat,
              const std::vector<RegTree *> &trees) override;
  // UpdatePredictionCache not supported.
};

}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_TREE_UPDATER_GPU_H_
