/*!
 * Copyright 2017-2022 by XGBoost Contributors
 * \file updater_quantile_hist.h
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Chen, Egor Smirnov
 */
#ifndef XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_
#define XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_

#include <rabit/rabit.h>
#include <xgboost/tree_updater.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xgboost/data.h"
#include "xgboost/json.h"

#include "hist/evaluate_splits.h"
#include "hist/histogram.h"
#include "hist/expand_entry.h"
#include "hist/param.h"

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
class HistRowPartitioner {
  // heuristically chosen block size of parallel partitioning
  static constexpr size_t kPartitionBlockSize = 2048;
  // worker class that partition a block of rows
  common::PartitionBuilder<kPartitionBlockSize> partition_builder_;
  // storage for row index
  common::RowSetCollection row_set_collection_;

  /**
   * \brief Turn split values into discrete bin indices.
   */
  static void FindSplitConditions(const std::vector<CPUExpandEntry>& nodes, const RegTree& tree,
                                  const GHistIndexMatrix& gmat,
                                  std::vector<int32_t>* split_conditions);
  /**
   * \brief Update the row set for new splits specifed by nodes.
   */
  void AddSplitsToRowSet(const std::vector<CPUExpandEntry>& nodes, RegTree const* p_tree);

 public:
  bst_row_t base_rowid = 0;

 public:
  HistRowPartitioner(size_t n_samples, size_t base_rowid, int32_t n_threads) {
    row_set_collection_.Clear();
    const size_t block_size = n_samples / n_threads + !!(n_samples % n_threads);
    dmlc::OMPException exc;
    std::vector<size_t>& row_indices = *row_set_collection_.Data();
    row_indices.resize(n_samples);
    size_t* p_row_indices = row_indices.data();
    // parallel initialization o f row indices. (std::iota)
#pragma omp parallel num_threads(n_threads)
    {
      exc.Run([&]() {
        const size_t tid = omp_get_thread_num();
        const size_t ibegin = tid * block_size;
        const size_t iend = std::min(static_cast<size_t>(ibegin + block_size), n_samples);
        for (size_t i = ibegin; i < iend; ++i) {
          p_row_indices[i] = i + base_rowid;
        }
      });
    }
    row_set_collection_.Init();
    this->base_rowid = base_rowid;
  }

  template <bool any_missing, bool any_cat>
  void UpdatePosition(GenericParameter const* ctx, GHistIndexMatrix const& gmat,
                      common::ColumnMatrix const& column_matrix,
                      std::vector<CPUExpandEntry> const& nodes, RegTree const* p_tree) {
    // 1. Find split condition for each split
    const size_t n_nodes = nodes.size();
    std::vector<int32_t> split_conditions;
    FindSplitConditions(nodes, *p_tree, gmat, &split_conditions);
    // 2.1 Create a blocked space of size SUM(samples in each node)
    common::BlockedSpace2d space(
        n_nodes,
        [&](size_t node_in_set) {
          int32_t nid = nodes[node_in_set].nid;
          return row_set_collection_[nid].Size();
        },
        kPartitionBlockSize);
    // 2.2 Initialize the partition builder
    // allocate buffers for storage intermediate results by each thread
    partition_builder_.Init(space.Size(), n_nodes, [&](size_t node_in_set) {
      const int32_t nid = nodes[node_in_set].nid;
      const size_t size = row_set_collection_[nid].Size();
      const size_t n_tasks = size / kPartitionBlockSize + !!(size % kPartitionBlockSize);
      return n_tasks;
    });
    CHECK_EQ(base_rowid, gmat.base_rowid);
    // 2.3 Split elements of row_set_collection_ to left and right child-nodes for each node
    // Store results in intermediate buffers from partition_builder_
    common::ParallelFor2d(space, ctx->Threads(), [&](size_t node_in_set, common::Range1d r) {
      size_t begin = r.begin();
      const int32_t nid = nodes[node_in_set].nid;
      const size_t task_id = partition_builder_.GetTaskIdx(node_in_set, begin);
      partition_builder_.AllocateForTask(task_id);
      switch (column_matrix.GetTypeSize()) {
        case common::kUint8BinsTypeSize:
          partition_builder_.template Partition<uint8_t, any_missing, any_cat>(
              node_in_set, nid, r, split_conditions[node_in_set], gmat, column_matrix, *p_tree,
              row_set_collection_[nid].begin);
          break;
        case common::kUint16BinsTypeSize:
          partition_builder_.template Partition<uint16_t, any_missing, any_cat>(
              node_in_set, nid, r, split_conditions[node_in_set], gmat, column_matrix, *p_tree,
              row_set_collection_[nid].begin);
          break;
        case common::kUint32BinsTypeSize:
          partition_builder_.template Partition<uint32_t, any_missing, any_cat>(
              node_in_set, nid, r, split_conditions[node_in_set], gmat, column_matrix, *p_tree,
              row_set_collection_[nid].begin);
          break;
        default:
          // no default behavior
          CHECK(false) << column_matrix.GetTypeSize();
      }
    });
    // 3. Compute offsets to copy blocks of row-indexes
    // from partition_builder_ to row_set_collection_
    partition_builder_.CalculateRowOffsets();

    // 4. Copy elements from partition_builder_ to row_set_collection_ back
    // with updated row-indexes for each tree-node
    common::ParallelFor2d(space, ctx->Threads(), [&](size_t node_in_set, common::Range1d r) {
      const int32_t nid = nodes[node_in_set].nid;
      partition_builder_.MergeToArray(node_in_set, r.begin(),
                                      const_cast<size_t*>(row_set_collection_[nid].begin));
    });
    // 5. Add info about splits into row_set_collection_
    AddSplitsToRowSet(nodes, p_tree);
  }

  void UpdatePosition(GenericParameter const* ctx, GHistIndexMatrix const& page,
                      std::vector<CPUExpandEntry> const& applied, RegTree const* p_tree) {
    auto const& column_matrix = page.Transpose();
    if (page.cut.HasCategorical()) {
      if (column_matrix.AnyMissing()) {
        this->template UpdatePosition<true, true>(ctx, page, column_matrix, applied, p_tree);
      } else {
        this->template UpdatePosition<false, true>(ctx, page, column_matrix, applied, p_tree);
      }
    } else {
      if (column_matrix.AnyMissing()) {
        this->template UpdatePosition<true, false>(ctx, page, column_matrix, applied, p_tree);
      } else {
        this->template UpdatePosition<false, false>(ctx, page, column_matrix, applied, p_tree);
      }
    }
  }

  auto const& Partitions() const { return row_set_collection_; }
  size_t Size() const {
    return std::distance(row_set_collection_.begin(), row_set_collection_.end());
  }
  auto& operator[](bst_node_t nidx) { return row_set_collection_[nidx]; }
  auto const& operator[](bst_node_t nidx) const { return row_set_collection_[nidx]; }
};

inline BatchParam HistBatch(TrainParam const& param) {
  return {param.max_bin, param.sparse_threshold};
}

/*! \brief construct a tree using quantized feature values */
class QuantileHistMaker: public TreeUpdater {
 public:
  explicit QuantileHistMaker(ObjInfo task) : task_{task} {}
  void Configure(const Args& args) override;

  void Update(HostDeviceVector<GradientPair>* gpair,
              DMatrix* dmat,
              const std::vector<RegTree*>& trees) override;

  bool UpdatePredictionCache(const DMatrix *data,
                             linalg::VectorView<float> out_preds) override;

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("train_param"), &this->param_);
    try {
      FromJson(config.at("cpu_hist_train_param"), &this->hist_maker_param_);
    } catch (std::out_of_range&) {
      // XGBoost model is from 1.1.x, so 'cpu_hist_train_param' is missing.
      // We add this compatibility check because it's just recently that we (developers) began
      // persuade R users away from using saveRDS() for model serialization. Hopefully, one day,
      // everyone will be using xgb.save().
      LOG(WARNING)
        << "Attempted to load internal configuration for a model file that was generated "
        << "by a previous version of XGBoost. A likely cause for this warning is that the model "
        << "was saved with saveRDS() in R or pickle.dump() in Python. We strongly ADVISE AGAINST "
        << "using saveRDS() or pickle.dump() so that the model remains accessible in current and "
        << "upcoming XGBoost releases. Please use xgb.save() instead to preserve models for the "
        << "long term. For more details and explanation, see "
        << "https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html";
      this->hist_maker_param_.UpdateAllowUnknown(Args{});
    }
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["train_param"] = ToJson(param_);
    out["cpu_hist_train_param"] = ToJson(hist_maker_param_);
  }

  char const* Name() const override {
    return "grow_quantile_histmaker";
  }

 protected:
  CPUHistMakerTrainParam hist_maker_param_;
  // training parameter
  TrainParam param_;

  // actual builder that runs the algorithm
  template<typename GradientSumT>
  struct Builder {
   public:
    using GradientPairT = xgboost::detail::GradientPairInternal<GradientSumT>;
    // constructor
    explicit Builder(const size_t n_trees, const TrainParam& param, DMatrix const* fmat,
                     ObjInfo task, GenericParameter const* ctx)
        : n_trees_(n_trees),
          param_(param),
          p_last_fmat_(fmat),
          histogram_builder_{new HistogramBuilder<GradientSumT, CPUExpandEntry>},
          task_{task},
          ctx_{ctx},
          monitor_{std::make_unique<common::Monitor>()} {
      monitor_->Init("Quantile::Builder");
    }
    // update one tree, growing
    void UpdateTree(HostDeviceVector<GradientPair>* gpair, DMatrix* p_fmat, RegTree* p_tree);

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

    void ExpandTree(DMatrix* p_fmat, RegTree* p_tree, const std::vector<GradientPair>& gpair_h);

   private:
    const size_t n_trees_;
    const TrainParam& param_;
    std::shared_ptr<common::ColumnSampler> column_sampler_{
        std::make_shared<common::ColumnSampler>()};

    std::vector<GradientPair> gpair_local_;

    std::unique_ptr<HistEvaluator<GradientSumT, CPUExpandEntry>> evaluator_;
    std::vector<HistRowPartitioner> partitioner_;

    // back pointers to tree and data matrix
    const RegTree* p_last_tree_{nullptr};
    DMatrix const* const p_last_fmat_;

    std::unique_ptr<HistogramBuilder<GradientSumT, CPUExpandEntry>> histogram_builder_;
    ObjInfo task_;
    // Context for number of threads
    GenericParameter const* ctx_;

    std::unique_ptr<common::Monitor> monitor_;
  };

 protected:
  std::unique_ptr<Builder<float>> float_builder_;
  std::unique_ptr<Builder<double>> double_builder_;
  ObjInfo task_;
};
}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_
