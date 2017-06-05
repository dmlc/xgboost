/*!
 * Copyright 2016 Rory Mitchell
 */
#include <xgboost/tree_updater.h>
#include <vector>
#include "../../../src/common/random.h"
#include "../../../src/common/sync.h"
#include "../../../src/tree/param.h"
#include "exact/gpu_builder.cuh"
#include "gpu_hist_builder.cuh"

namespace xgboost {
namespace tree {
DMLC_REGISTRY_FILE_TAG(updater_gpumaker);

/*! \brief column-wise update to construct a tree */
template <typename TStats>
class GPUMaker : public TreeUpdater {
 public:
  void Init(
      const std::vector<std::pair<std::string, std::string>>& args) override {
    param.InitAllowUnknown(args);
    builder.Init(param);
  }

  void Update(const std::vector<bst_gpair>& gpair, DMatrix* dmat,
              const std::vector<RegTree*>& trees) override {
    TStats::CheckInfo(dmat->info());
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();
    builder.UpdateParam(param);

    try {
      // build tree
      for (size_t i = 0; i < trees.size(); ++i) {
        builder.Update(gpair, dmat, trees[i]);
      }
    } catch (const std::exception& e) {
      LOG(FATAL) << "GPU plugin exception: " << e.what() << std::endl;
    }
    param.learning_rate = lr;
  }

 protected:
  // training parameter
  TrainParam param;
  exact::GPUBuilder<int16_t> builder;
};

template <typename TStats>
class GPUHistMaker : public TreeUpdater {
 public:
  void Init(
      const std::vector<std::pair<std::string, std::string>>& args) override {
    param.InitAllowUnknown(args);
    builder.Init(param);
  }

  void Update(const std::vector<bst_gpair>& gpair, DMatrix* dmat,
              const std::vector<RegTree*>& trees) override {
    TStats::CheckInfo(dmat->info());
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();
    builder.UpdateParam(param);
    // build tree
    try {
      for (size_t i = 0; i < trees.size(); ++i) {
        builder.Update(gpair, dmat, trees[i]);
      }
    } catch (const std::exception& e) {
      LOG(FATAL) << "GPU plugin exception: " << e.what() << std::endl;
    }
    param.learning_rate = lr;
  }

  bool UpdatePredictionCache(const DMatrix* data,
    std::vector<bst_float>* out_preds) override {
    return builder.UpdatePredictionCache(data, out_preds);
  }

 protected:
  // training parameter
  TrainParam param;
  GPUHistBuilder builder;
};

XGBOOST_REGISTER_TREE_UPDATER(GPUMaker, "grow_gpu")
    .describe("Grow tree with GPU.")
    .set_body([]() { return new GPUMaker<GradStats>(); });

XGBOOST_REGISTER_TREE_UPDATER(GPUHistMaker, "grow_gpu_hist")
    .describe("Grow tree with GPU.")
    .set_body([]() { return new GPUHistMaker<GradStats>(); });
}  // namespace tree
}  // namespace xgboost
