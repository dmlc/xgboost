/*!
 * Copyright 2017-2022 by XGBoost Contributors
 * \file updater_quantile_hist.h
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Chen, Egor Smirnov
 */
#ifndef XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_
#define XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_

#include <xgboost/tree_updater.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/json.h"

#include "hist/evaluate_splits.h"
#include "hist/histogram.h"
#include "hist/expand_entry.h"

#include "common_row_partitioner.h"
#include "constraints.h"
#include "./param.h"
#include "./driver.h"
#include "../common/random.h"
#include "../common/timer.h"
#include "../common/hist_util.h"
#include "../common/row_set.h"
#include "../common/partition_builder.h"
#include "../common/column_matrix.h"

namespace xgboost {
struct RandomReplace {
 public:
  // similar value as for minstd_rand
  static constexpr uint64_t kBase = 16807;
  static constexpr uint64_t kMod = static_cast<uint64_t>(1) << 63;

  using EngineT = std::linear_congruential_engine<uint64_t, kBase, 0, kMod>;

  /*
    Right-to-left binary method: https://en.wikipedia.org/wiki/Modular_exponentiation
  */
  static uint64_t SimpleSkip(uint64_t exponent, uint64_t initial_seed,
                             uint64_t base, uint64_t mod) {
    CHECK_LE(exponent, mod);
    uint64_t result = 1;
    while (exponent > 0) {
      if (exponent % 2 == 1) {
        result = (result * base) % mod;
      }
      base = (base * base) % mod;
      exponent = exponent >> 1;
    }
    // with result we can now find the new seed
    return (result * initial_seed) % mod;
  }

  template<typename Condition, typename ContainerData>
  static void MakeIf(Condition condition, const typename ContainerData::value_type replace_value,
                     const uint64_t initial_seed, const size_t ibegin,
                     const size_t iend, ContainerData* gpair) {
    ContainerData& gpair_ref = *gpair;
    const uint64_t displaced_seed = SimpleSkip(ibegin, initial_seed, kBase, kMod);
    EngineT eng(displaced_seed);
    for (size_t i = ibegin; i < iend; ++i) {
      if (condition(i, eng)) {
        gpair_ref[i] = replace_value;
      }
    }
  }
};

namespace tree {
inline BatchParam HistBatch(TrainParam const& param) {
  return {param.max_bin, param.sparse_threshold};
}

/*! \brief construct a tree using quantized feature values */
class QuantileHistMaker: public TreeUpdater {
 public:
  explicit QuantileHistMaker(GenericParameter const* ctx, ObjInfo task)
      : TreeUpdater(ctx), task_{task} {}
  void Configure(const Args& args) override;

  void Update(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat,
              common::Span<HostDeviceVector<bst_node_t>> out_position,
              const std::vector<RegTree*>& trees) override;

  bool UpdatePredictionCache(const DMatrix *data,
                             linalg::VectorView<float> out_preds) override;

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("train_param"), &this->param_);
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["train_param"] = ToJson(param_);
  }

  char const* Name() const override {
    return "grow_quantile_histmaker";
  }

  bool HasNodePosition() const override { return true; }

 protected:
  // training parameter
  TrainParam param_;

  // actual builder that runs the algorithm
  struct Builder {
   public:
    // constructor
    explicit Builder(const size_t n_trees, const TrainParam& param, DMatrix const* fmat,
                     ObjInfo task, GenericParameter const* ctx)
        : n_trees_(n_trees),
          param_(param),
          p_last_fmat_(fmat),
          histogram_builder_{new HistogramBuilder<CPUExpandEntry>},
          task_{task},
          ctx_{ctx},
          monitor_{std::make_unique<common::Monitor>()} {
      monitor_->Init("Quantile::Builder");
    }
    // update one tree, growing
    void UpdateTree(HostDeviceVector<GradientPair>* gpair, DMatrix* p_fmat, RegTree* p_tree,
                    HostDeviceVector<bst_node_t>* p_out_position);

    bool UpdatePredictionCache(DMatrix const* data, linalg::VectorView<float> out_preds) const;

   private:
    // initialize temp data structure
    void InitData(DMatrix* fmat, const RegTree& tree, std::vector<GradientPair>* gpair);

    size_t GetNumberOfTrees();

    void InitSampling(const DMatrix& fmat, std::vector<GradientPair>* gpair);

    CPUExpandEntry InitRoot(DMatrix* p_fmat, RegTree* p_tree,
                            const std::vector<GradientPair>& gpair_h);

    void BuildHistogram(DMatrix* p_fmat, RegTree* p_tree,
                        std::vector<CPUExpandEntry> const& valid_candidates,
                        std::vector<GradientPair> const& gpair);

    void LeafPartition(RegTree const& tree, common::Span<GradientPair const> gpair,
                       std::vector<bst_node_t>* p_out_position);

    void ExpandTree(DMatrix* p_fmat, RegTree* p_tree, const std::vector<GradientPair>& gpair_h,
                    HostDeviceVector<bst_node_t>* p_out_position);

   private:
    const size_t n_trees_;
    const TrainParam& param_;
    std::shared_ptr<common::ColumnSampler> column_sampler_{
        std::make_shared<common::ColumnSampler>()};

    std::vector<GradientPair> gpair_local_;

    std::unique_ptr<HistEvaluator<CPUExpandEntry>> evaluator_;
    std::vector<CommonRowPartitioner> partitioner_;

    // back pointers to tree and data matrix
    const RegTree* p_last_tree_{nullptr};
    DMatrix const* const p_last_fmat_;

    std::unique_ptr<HistogramBuilder<CPUExpandEntry>> histogram_builder_;
    ObjInfo task_;
    // Context for number of threads
    Context const* ctx_;

    std::unique_ptr<common::Monitor> monitor_;
  };

 protected:
  std::unique_ptr<Builder> pimpl_;
  ObjInfo task_;
};
}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_
