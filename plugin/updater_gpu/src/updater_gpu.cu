/*!
 * Copyright 2017 XGBoost contributors
 */
#include <xgboost/tree_updater.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "../../../src/common/random.h"
#include "../../../src/common/sync.h"
#include "../../../src/tree/param.h"
#include "exact/gpu_builder.cuh"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_gpu);

class GPUMaker : public TreeUpdater {
 protected:
  TrainParam param;
  std::unique_ptr<exact::GPUBuilder> builder;

 public:
  GPUMaker() : builder(new exact::GPUBuilder()) {}
  void Init(
      const std::vector<std::pair<std::string, std::string>>& args) override {
    param.InitAllowUnknown(args);
    builder->Init(param);
  }


  void Update(const std::vector<bst_gpair>& gpair, DMatrix* dmat,
              const std::vector<RegTree*>& trees) override {
    GradStats::CheckInfo(dmat->info());
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();
    builder->UpdateParam(param);

    try {
      // build tree
      for (size_t i = 0; i < trees.size(); ++i) {
        builder->Update(gpair, dmat, trees[i]);
      }
    } catch (const std::exception& e) {
      LOG(FATAL) << "GPU plugin exception: " << e.what() << std::endl;
    }
    param.learning_rate = lr;
  }
};


XGBOOST_REGISTER_TREE_UPDATER(GPUMaker, "grow_gpu")
    .describe("Grow tree with GPU.")
    .set_body([]() { return new GPUMaker(); });

}  // namespace tree
}  // namespace xgboost
