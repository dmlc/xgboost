/*!
 * Copyright 2017-2022 XGBoost contributors
 */
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <xgboost/tree_updater.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <limits>
#include <utility>
#include <vector>

#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/generic_parameters.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/parameter.h"
#include "xgboost/span.h"
#include "xgboost/json.h"

#include "../common/io.h"
#include "../common/device_helpers.cuh"
#include "../common/hist_util.h"
#include "../common/bitfield.h"
#include "../common/timer.h"
#include "../common/categorical.h"
#include "../data/ellpack_page.cuh"

#include "param.h"
#include "driver.h"
#include "updater_gpu_common.cuh"
#include "split_evaluator.h"
#include "constraints.cuh"
#include "gpu_hist/feature_groups.cuh"
#include "gpu_hist/gradient_based_sampler.cuh"
#include "gpu_hist/row_partitioner.cuh"
#include "gpu_hist/histogram.cuh"
#include "gpu_hist/evaluate_splits.cuh"
#include "gpu_hist/expand_entry.cuh"
#include "xgboost/task.h"
#include "xgboost/tree_model.h"

namespace xgboost {
namespace tree {
#if !defined(GTEST_TEST)
DMLC_REGISTRY_FILE_TAG(updater_gpu_hist);
#endif  // !defined(GTEST_TEST)

// training parameters specific to this algorithm
struct GPUHistMakerTrainParam
    : public XGBoostParameter<GPUHistMakerTrainParam> {
  bool debug_synchronize;
  // declare parameters
  DMLC_DECLARE_PARAMETER(GPUHistMakerTrainParam) {
    DMLC_DECLARE_FIELD(debug_synchronize).set_default(false).describe(
        "Check if all distributed tree are identical after tree construction.");
  }
};
#if !defined(GTEST_TEST)
DMLC_REGISTER_PARAMETER(GPUHistMakerTrainParam);
#endif  // !defined(GTEST_TEST)

/**
 * \struct  DeviceHistogram
 *
 * \summary Data storage for node histograms on device. Automatically expands.
 *
 * \tparam GradientSumT      histogram entry type.
 * \tparam kStopGrowingSize  Do not grow beyond this size
 *
 * \author  Rory
 * \date    28/07/2018
 */
template <typename GradientSumT, size_t kStopGrowingSize = 1 << 26>
class DeviceHistogram {
 private:
  /*! \brief Map nidx to starting index of its histogram. */
  std::map<int, size_t> nidx_map_;
  dh::device_vector<typename GradientSumT::ValueT> data_;
  int n_bins_;
  int device_id_;
  static constexpr size_t kNumItemsInGradientSum =
      sizeof(GradientSumT) / sizeof(typename GradientSumT::ValueT);
  static_assert(kNumItemsInGradientSum == 2,
                "Number of items in gradient type should be 2.");

 public:
  void Init(int device_id, int n_bins) {
    this->n_bins_ = n_bins;
    this->device_id_ = device_id;
  }

  void Reset() {
    auto d_data = data_.data().get();
    dh::LaunchN(data_.size(),
                [=] __device__(size_t idx) { d_data[idx] = 0.0f; });
    nidx_map_.clear();
  }
  bool HistogramExists(int nidx) const {
    return nidx_map_.find(nidx) != nidx_map_.cend();
  }
  int Bins() const {
    return n_bins_;
  }
  size_t HistogramSize() const {
    return n_bins_ * kNumItemsInGradientSum;
  }

  dh::device_vector<typename GradientSumT::ValueT>& Data() {
    return data_;
  }

  void AllocateHistogram(int nidx) {
    if (HistogramExists(nidx)) return;
    // Number of items currently used in data
    const size_t used_size = nidx_map_.size() * HistogramSize();
    const size_t new_used_size = used_size + HistogramSize();
    if (data_.size() >= kStopGrowingSize) {
      // Recycle histogram memory
      if (new_used_size <= data_.size()) {
        // no need to remove old node, just insert the new one.
        nidx_map_[nidx] = used_size;
        // memset histogram size in bytes
      } else {
        std::pair<int, size_t> old_entry = *nidx_map_.begin();
        nidx_map_.erase(old_entry.first);
        nidx_map_[nidx] = old_entry.second;
      }
      // Zero recycled memory
      auto d_data = data_.data().get() + nidx_map_[nidx];
      dh::LaunchN(n_bins_ * 2,
                  [=] __device__(size_t idx) { d_data[idx] = 0.0f; });
    } else {
      // Append new node histogram
      nidx_map_[nidx] = used_size;
      // Check there is enough memory for another histogram node
      if (data_.size() < new_used_size + HistogramSize()) {
        size_t new_required_memory =
            std::max(data_.size() * 2, HistogramSize());
        data_.resize(new_required_memory);
      }
    }

    CHECK_GE(data_.size(), nidx_map_.size() * HistogramSize());
  }

  /**
   * \summary   Return pointer to histogram memory for a given node.
   * \param nidx    Tree node index.
   * \return    hist pointer.
   */
  common::Span<GradientSumT> GetNodeHistogram(int nidx) {
    CHECK(this->HistogramExists(nidx));
    auto ptr = data_.data().get() + nidx_map_.at(nidx);
    return common::Span<GradientSumT>(
        reinterpret_cast<GradientSumT*>(ptr), n_bins_);
  }
};

// Manage memory for a single GPU
template <typename GradientSumT>
struct GPUHistMakerDevice {
 private:
  GPUHistEvaluator<GradientSumT> evaluator_;
  Context const* ctx_;

 public:
  EllpackPageImpl const* page;
  common::Span<FeatureType const> feature_types;
  BatchParam batch_param;

  std::unique_ptr<RowPartitioner> row_partitioner;
  DeviceHistogram<GradientSumT> hist{};

  dh::caching_device_vector<GradientPair> d_gpair;  // storage for gpair;
  common::Span<GradientPair> gpair;

  dh::caching_device_vector<int> monotone_constraints;

  /*! \brief Sum gradient for each node. */
  std::vector<GradientPairPrecise> node_sum_gradients;

  TrainParam param;

  HistRounding<GradientSumT> histogram_rounding;

  dh::PinnedMemory pinned;

  common::Monitor monitor;
  common::ColumnSampler column_sampler;
  FeatureInteractionConstraintDevice interaction_constraints;

  std::unique_ptr<GradientBasedSampler> sampler;

  std::unique_ptr<FeatureGroups> feature_groups;

  GPUHistMakerDevice(Context const* ctx, EllpackPageImpl const* _page,
                     common::Span<FeatureType const> _feature_types, bst_uint _n_rows,
                     TrainParam _param, uint32_t column_sampler_seed, uint32_t n_features,
                     BatchParam _batch_param)
      : evaluator_{_param, n_features, ctx->gpu_id},
        ctx_(ctx),
        page(_page),
        feature_types{_feature_types},
        param(std::move(_param)),
        column_sampler(column_sampler_seed),
        interaction_constraints(param, n_features),
        batch_param(std::move(_batch_param)) {
    sampler.reset(new GradientBasedSampler(page, _n_rows, batch_param, param.subsample,
                                           param.sampling_method));
    if (!param.monotone_constraints.empty()) {
      // Copy assigning an empty vector causes an exception in MSVC debug builds
      monotone_constraints = param.monotone_constraints;
    }
    node_sum_gradients.resize(param.MaxNodes());

    // Init histogram
    hist.Init(ctx_->gpu_id, page->Cuts().TotalBins());
    monitor.Init(std::string("GPUHistMakerDevice") + std::to_string(ctx_->gpu_id));
    feature_groups.reset(new FeatureGroups(page->Cuts(), page->is_dense,
                                           dh::MaxSharedMemoryOptin(ctx_->gpu_id),
                                           sizeof(GradientSumT)));
  }

  ~GPUHistMakerDevice() {  // NOLINT
    dh::safe_cuda(cudaSetDevice(ctx_->gpu_id));
  }

  // Reset values for each update iteration
  // Note that the column sampler must be passed by value because it is not
  // thread safe
  void Reset(HostDeviceVector<GradientPair>* dh_gpair, DMatrix* dmat, int64_t num_columns) {
    auto const& info = dmat->Info();
    this->column_sampler.Init(num_columns, info.feature_weights.HostVector(),
                              param.colsample_bynode, param.colsample_bylevel,
                              param.colsample_bytree);
    dh::safe_cuda(cudaSetDevice(ctx_->gpu_id));

    this->evaluator_.Reset(page->Cuts(), feature_types, dmat->Info().num_col_, param,
                           ctx_->gpu_id);

    this->interaction_constraints.Reset();
    std::fill(node_sum_gradients.begin(), node_sum_gradients.end(), GradientPairPrecise{});

    if (d_gpair.size() != dh_gpair->Size()) {
      d_gpair.resize(dh_gpair->Size());
    }
    dh::safe_cuda(cudaMemcpyAsync(
        d_gpair.data().get(), dh_gpair->ConstDevicePointer(),
        dh_gpair->Size() * sizeof(GradientPair), cudaMemcpyDeviceToDevice));
    auto sample = sampler->Sample(dh::ToSpan(d_gpair), dmat);
    page = sample.page;
    gpair = sample.gpair;

    histogram_rounding = CreateRoundingFactor<GradientSumT>(this->gpair);

    row_partitioner.reset();  // Release the device memory first before reallocating
    row_partitioner.reset(new RowPartitioner(ctx_->gpu_id,  sample.sample_rows));
    hist.Reset();
  }

  GPUExpandEntry EvaluateRootSplit(GradientPairPrecise root_sum, float weight) {
    int nidx = RegTree::kRoot;
    GPUTrainingParam gpu_param(param);
    auto sampled_features = column_sampler.GetFeatureSet(0);
    sampled_features->SetDevice(ctx_->gpu_id);
    common::Span<bst_feature_t> feature_set =
        interaction_constraints.Query(sampled_features->DeviceSpan(), nidx);
    auto matrix = page->GetDeviceAccessor(ctx_->gpu_id);
    EvaluateSplitInputs<GradientSumT> inputs{nidx,
                                             root_sum,
                                             gpu_param,
                                             feature_set,
                                             feature_types,
                                             matrix.feature_segments,
                                             matrix.gidx_fvalue_map,
                                             matrix.min_fvalue,
                                             hist.GetNodeHistogram(nidx)};
    auto split = this->evaluator_.EvaluateSingleSplit(inputs, weight);
    return split;
  }

  void EvaluateLeftRightSplits(GPUExpandEntry candidate, int left_nidx, int right_nidx,
                               const RegTree& tree,
                               common::Span<GPUExpandEntry> pinned_candidates_out) {
    dh::TemporaryArray<DeviceSplitCandidate> splits_out(2);
    GPUTrainingParam gpu_param(param);
    auto left_sampled_features = column_sampler.GetFeatureSet(tree.GetDepth(left_nidx));
    left_sampled_features->SetDevice(ctx_->gpu_id);
    common::Span<bst_feature_t> left_feature_set =
        interaction_constraints.Query(left_sampled_features->DeviceSpan(), left_nidx);
    auto right_sampled_features = column_sampler.GetFeatureSet(tree.GetDepth(right_nidx));
    right_sampled_features->SetDevice(ctx_->gpu_id);
    common::Span<bst_feature_t> right_feature_set =
        interaction_constraints.Query(right_sampled_features->DeviceSpan(), left_nidx);
    auto matrix = page->GetDeviceAccessor(ctx_->gpu_id);

    EvaluateSplitInputs<GradientSumT> left{left_nidx,
                                           candidate.split.left_sum,
                                           gpu_param,
                                           left_feature_set,
                                           feature_types,
                                           matrix.feature_segments,
                                           matrix.gidx_fvalue_map,
                                           matrix.min_fvalue,
                                           hist.GetNodeHistogram(left_nidx)};
    EvaluateSplitInputs<GradientSumT> right{right_nidx,
                                            candidate.split.right_sum,
                                            gpu_param,
                                            right_feature_set,
                                            feature_types,
                                            matrix.feature_segments,
                                            matrix.gidx_fvalue_map,
                                            matrix.min_fvalue,
                                            hist.GetNodeHistogram(right_nidx)};

    dh::TemporaryArray<GPUExpandEntry> entries(2);
    this->evaluator_.EvaluateSplits(candidate, left, right, dh::ToSpan(entries));
    dh::safe_cuda(cudaMemcpyAsync(pinned_candidates_out.data(), entries.data().get(),
                                  sizeof(GPUExpandEntry) * entries.size(), cudaMemcpyDeviceToHost));
  }

  void BuildHist(int nidx) {
    hist.AllocateHistogram(nidx);
    auto d_node_hist = hist.GetNodeHistogram(nidx);
    auto d_ridx = row_partitioner->GetRows(nidx);
    BuildGradientHistogram(page->GetDeviceAccessor(ctx_->gpu_id),
                           feature_groups->DeviceAccessor(ctx_->gpu_id), gpair,
                           d_ridx, d_node_hist, histogram_rounding);
  }

  void SubtractionTrick(int nidx_parent, int nidx_histogram,
                        int nidx_subtraction) {
    auto d_node_hist_parent = hist.GetNodeHistogram(nidx_parent);
    auto d_node_hist_histogram = hist.GetNodeHistogram(nidx_histogram);
    auto d_node_hist_subtraction = hist.GetNodeHistogram(nidx_subtraction);

    dh::LaunchN(page->Cuts().TotalBins(), [=] __device__(size_t idx) {
      d_node_hist_subtraction[idx] =
          d_node_hist_parent[idx] - d_node_hist_histogram[idx];
    });
  }

  bool CanDoSubtractionTrick(int nidx_parent, int nidx_histogram, int nidx_subtraction) {
    // Make sure histograms are already allocated
    hist.AllocateHistogram(nidx_subtraction);
    return hist.HistogramExists(nidx_histogram) && hist.HistogramExists(nidx_parent);
  }

  void UpdatePosition(const GPUExpandEntry &e, RegTree* p_tree) {
    RegTree::Node split_node = (*p_tree)[e.nid];
    auto split_type = p_tree->NodeSplitType(e.nid);
    auto d_matrix = page->GetDeviceAccessor(ctx_->gpu_id);
    auto node_cats = e.split.split_cats.Bits();

    row_partitioner->UpdatePosition(
        e.nid, split_node.LeftChild(), split_node.RightChild(),
        [=] __device__(bst_uint ridx) {
          // given a row index, returns the node id it belongs to
          bst_float cut_value =
              d_matrix.GetFvalue(ridx, split_node.SplitIndex());
          // Missing value
          bst_node_t new_position = 0;
          if (isnan(cut_value)) {
            new_position = split_node.DefaultChild();
          } else {
            bool go_left = true;
            if (split_type == FeatureType::kCategorical) {
              go_left = common::Decision<false>(node_cats, cut_value, split_node.DefaultLeft());
            } else {
              go_left = cut_value <= split_node.SplitCond();
            }
            if (go_left) {
              new_position = split_node.LeftChild();
            } else {
              new_position = split_node.RightChild();
            }
          }
          return new_position;
        });
  }

  // After tree update is finished, update the position of all training
  // instances to their final leaf. This information is used later to update the
  // prediction cache
  void FinalisePosition(RegTree const* p_tree, DMatrix* p_fmat, ObjInfo task,
                        HostDeviceVector<bst_node_t>* p_out_position) {
    dh::TemporaryArray<RegTree::Node> d_nodes(p_tree->GetNodes().size());
    dh::safe_cuda(cudaMemcpyAsync(d_nodes.data().get(), p_tree->GetNodes().data(),
                                  d_nodes.size() * sizeof(RegTree::Node),
                                  cudaMemcpyHostToDevice));
    auto const& h_split_types = p_tree->GetSplitTypes();
    auto const& categories = p_tree->GetSplitCategories();
    auto const& categories_segments = p_tree->GetSplitCategoriesPtr();

    dh::caching_device_vector<FeatureType> d_split_types;
    dh::caching_device_vector<uint32_t> d_categories;
    dh::caching_device_vector<RegTree::Segment> d_categories_segments;

    if (!categories.empty()) {
      dh::CopyToD(h_split_types, &d_split_types);
      dh::CopyToD(categories, &d_categories);
      dh::CopyToD(categories_segments, &d_categories_segments);
    }

    if (row_partitioner->GetRows().size() != p_fmat->Info().num_row_) {
      row_partitioner.reset();  // Release the device memory first before reallocating
      row_partitioner.reset(new RowPartitioner(ctx_->gpu_id, p_fmat->Info().num_row_));
    }
    if (task.UpdateTreeLeaf() && !p_fmat->SingleColBlock() && param.subsample != 1.0) {
      // see comment in the `FinalisePositionInPage`.
      LOG(FATAL) << "Current objective function can not be used with subsampled external memory.";
    }
    if (page->n_rows == p_fmat->Info().num_row_) {
      FinalisePositionInPage(page, dh::ToSpan(d_nodes), dh::ToSpan(d_split_types),
                             dh::ToSpan(d_categories), dh::ToSpan(d_categories_segments), task,
                             p_out_position);
    } else {
      for (auto const& batch : p_fmat->GetBatches<EllpackPage>(batch_param)) {
        FinalisePositionInPage(batch.Impl(), dh::ToSpan(d_nodes), dh::ToSpan(d_split_types),
                               dh::ToSpan(d_categories), dh::ToSpan(d_categories_segments), task,
                               p_out_position);
      }
    }
  }

  void FinalisePositionInPage(EllpackPageImpl const *page,
                              const common::Span<RegTree::Node> d_nodes,
                              common::Span<FeatureType const> d_feature_types,
                              common::Span<uint32_t const> categories,
                              common::Span<RegTree::Segment> categories_segments,
                              ObjInfo task,
                              HostDeviceVector<bst_node_t>* p_out_position) {
    auto d_matrix = page->GetDeviceAccessor(ctx_->gpu_id);
    auto d_gpair = this->gpair;
    row_partitioner->FinalisePosition(
        ctx_, task, p_out_position,
        [=] __device__(size_t row_id, int position) {
          // What happens if user prune the tree?
          if (!d_matrix.IsInRange(row_id)) {
            return RowPartitioner::kIgnoredTreePosition;
          }
          auto node = d_nodes[position];

          while (!node.IsLeaf()) {
            bst_float element = d_matrix.GetFvalue(row_id, node.SplitIndex());
            // Missing value
            if (isnan(element)) {
              position = node.DefaultChild();
            } else {
              bool go_left = true;
              if (common::IsCat(d_feature_types, position)) {
                auto node_cats =
                    categories.subspan(categories_segments[position].beg,
                                       categories_segments[position].size);
                go_left = common::Decision<false>(node_cats, element, node.DefaultLeft());
              } else {
                go_left = element <= node.SplitCond();
              }
              if (go_left) {
                position = node.LeftChild();
              } else {
                position = node.RightChild();
              }
            }
            node = d_nodes[position];
          }

          return position;
        },
        [d_gpair] __device__(size_t ridx) {
          // FIXME(jiamingy): Doesn't work when sampling is used with external memory as
          // the sampler compacts the gradient vector.
          return d_gpair[ridx].GetHess() - .0f == 0.f;
        });
  }

  void UpdatePredictionCache(linalg::VectorView<float> out_preds_d, RegTree const* p_tree) {
    CHECK(p_tree);
    dh::safe_cuda(cudaSetDevice(ctx_->gpu_id));
    CHECK_EQ(out_preds_d.DeviceIdx(), ctx_->gpu_id);
    auto d_ridx = row_partitioner->GetRows();

    GPUTrainingParam param_d(param);
    dh::TemporaryArray<GradientPairPrecise> device_node_sum_gradients(node_sum_gradients.size());

    dh::safe_cuda(cudaMemcpyAsync(device_node_sum_gradients.data().get(), node_sum_gradients.data(),
                                  sizeof(GradientPairPrecise) * node_sum_gradients.size(),
                                  cudaMemcpyHostToDevice));
    auto d_position = row_partitioner->GetPosition();
    auto d_node_sum_gradients = device_node_sum_gradients.data().get();
    auto tree_evaluator = evaluator_.GetEvaluator();

    auto const& h_nodes = p_tree->GetNodes();
    dh::caching_device_vector<RegTree::Node> nodes(h_nodes.size());
    dh::safe_cuda(cudaMemcpyAsync(nodes.data().get(), h_nodes.data(),
                                  h_nodes.size() * sizeof(RegTree::Node), cudaMemcpyHostToDevice));
    auto d_nodes = dh::ToSpan(nodes);
    dh::LaunchN(d_ridx.size(), [=] XGBOOST_DEVICE(size_t idx) mutable {
      bst_node_t nidx = d_position[idx];
      auto weight = d_nodes[nidx].LeafValue();
      out_preds_d(d_ridx[idx]) += weight;
    });
    row_partitioner.reset();
  }

  void AllReduceHist(int nidx, dh::AllReducer* reducer) {
    monitor.Start("AllReduce");
    auto d_node_hist = hist.GetNodeHistogram(nidx).data();
    reducer->AllReduceSum(
        reinterpret_cast<typename GradientSumT::ValueT*>(d_node_hist),
        reinterpret_cast<typename GradientSumT::ValueT*>(d_node_hist),
        page->Cuts().TotalBins() * (sizeof(GradientSumT) / sizeof(typename GradientSumT::ValueT)));

    monitor.Stop("AllReduce");
  }

  /**
   * \brief Build GPU local histograms for the left and right child of some parent node
   */
  void BuildHistLeftRight(const GPUExpandEntry &candidate, int nidx_left,
        int nidx_right, dh::AllReducer* reducer) {
    auto build_hist_nidx = nidx_left;
    auto subtraction_trick_nidx = nidx_right;

    // Decide whether to build the left histogram or right histogram
    // Use sum of Hessian as a heuristic to select node with fewest training instances
    bool fewer_right = candidate.split.right_sum.GetHess() < candidate.split.left_sum.GetHess();
    if (fewer_right) {
      std::swap(build_hist_nidx, subtraction_trick_nidx);
    }

    this->BuildHist(build_hist_nidx);
    this->AllReduceHist(build_hist_nidx, reducer);

    // Check whether we can use the subtraction trick to calculate the other
    bool do_subtraction_trick = this->CanDoSubtractionTrick(
        candidate.nid, build_hist_nidx, subtraction_trick_nidx);

    if (do_subtraction_trick) {
      // Calculate other histogram using subtraction trick
      this->SubtractionTrick(candidate.nid, build_hist_nidx,
                             subtraction_trick_nidx);
    } else {
      // Calculate other histogram manually
      this->BuildHist(subtraction_trick_nidx);
      this->AllReduceHist(subtraction_trick_nidx, reducer);
    }
  }

  void ApplySplit(const GPUExpandEntry& candidate, RegTree* p_tree) {
    RegTree& tree = *p_tree;

    // Sanity check - have we created a leaf with no training instances?
    if (!rabit::IsDistributed() && row_partitioner) {
      CHECK(row_partitioner->GetRows(candidate.nid).size() > 0)
          << "No training instances in this leaf!";
    }

    auto parent_sum = candidate.split.left_sum + candidate.split.right_sum;
    auto base_weight = candidate.base_weight;
    auto left_weight = candidate.left_weight * param.learning_rate;
    auto right_weight = candidate.right_weight * param.learning_rate;

    auto is_cat = candidate.split.is_cat;
    if (is_cat) {
      CHECK_LT(candidate.split.fvalue, std::numeric_limits<bst_cat_t>::max())
          << "Categorical feature value too large.";
      std::vector<uint32_t> split_cats;
      CHECK_GT(candidate.split.split_cats.Bits().size(), 0);
      auto h_cats = this->evaluator_.GetHostNodeCats(candidate.nid);
      auto max_cat = candidate.split.MaxCat();
      split_cats.resize(common::CatBitField::ComputeStorageSize(max_cat + 1), 0);
      CHECK_LE(split_cats.size(), h_cats.size());
      std::copy(h_cats.data(), h_cats.data() + split_cats.size(), split_cats.data());

      tree.ExpandCategorical(
          candidate.nid, candidate.split.findex, split_cats, candidate.split.dir == kLeftDir,
          base_weight, left_weight, right_weight, candidate.split.loss_chg, parent_sum.GetHess(),
          candidate.split.left_sum.GetHess(), candidate.split.right_sum.GetHess());
    } else {
      tree.ExpandNode(candidate.nid, candidate.split.findex, candidate.split.fvalue,
                      candidate.split.dir == kLeftDir, base_weight, left_weight, right_weight,
                      candidate.split.loss_chg, parent_sum.GetHess(),
                      candidate.split.left_sum.GetHess(), candidate.split.right_sum.GetHess());
    }
    evaluator_.ApplyTreeSplit(candidate, p_tree);

    node_sum_gradients[tree[candidate.nid].LeftChild()] = candidate.split.left_sum;
    node_sum_gradients[tree[candidate.nid].RightChild()] = candidate.split.right_sum;

    interaction_constraints.Split(candidate.nid, tree[candidate.nid].SplitIndex(),
                                  tree[candidate.nid].LeftChild(),
                                  tree[candidate.nid].RightChild());
  }

  GPUExpandEntry InitRoot(RegTree* p_tree, dh::AllReducer* reducer) {
    constexpr bst_node_t kRootNIdx = 0;
    dh::XGBCachingDeviceAllocator<char> alloc;
    auto gpair_it = dh::MakeTransformIterator<GradientPairPrecise>(
        dh::tbegin(gpair), [] __device__(auto const& gpair) { return GradientPairPrecise{gpair}; });
    GradientPairPrecise root_sum =
        dh::Reduce(thrust::cuda::par(alloc), gpair_it, gpair_it + gpair.size(),
                   GradientPairPrecise{}, thrust::plus<GradientPairPrecise>{});
    rabit::Allreduce<rabit::op::Sum, double>(reinterpret_cast<double*>(&root_sum), 2);

    this->BuildHist(kRootNIdx);
    this->AllReduceHist(kRootNIdx, reducer);

    // Remember root stats
    node_sum_gradients[kRootNIdx] = root_sum;
    p_tree->Stat(kRootNIdx).sum_hess = root_sum.GetHess();
    auto weight = CalcWeight(param, root_sum);
    p_tree->Stat(kRootNIdx).base_weight = weight;
    (*p_tree)[kRootNIdx].SetLeaf(param.learning_rate * weight);

    // Generate first split
    auto root_entry = this->EvaluateRootSplit(root_sum, weight);
    return root_entry;
  }

  void UpdateTree(HostDeviceVector<GradientPair>* gpair_all, DMatrix* p_fmat, ObjInfo task,
                  RegTree* p_tree, dh::AllReducer* reducer,
                  HostDeviceVector<bst_node_t>* p_out_position) {
    auto& tree = *p_tree;
    Driver<GPUExpandEntry> driver(static_cast<TrainParam::TreeGrowPolicy>(param.grow_policy));

    monitor.Start("Reset");
    this->Reset(gpair_all, p_fmat, p_fmat->Info().num_col_);
    monitor.Stop("Reset");

    monitor.Start("InitRoot");
    driver.Push({ this->InitRoot(p_tree, reducer) });
    monitor.Stop("InitRoot");

    auto num_leaves = 1;

    // The set of leaves that can be expanded asynchronously
    auto expand_set = driver.Pop();
    while (!expand_set.empty()) {
      auto new_candidates =
          pinned.GetSpan<GPUExpandEntry>(expand_set.size() * 2, GPUExpandEntry());

      for (auto i = 0ull; i < expand_set.size(); i++) {
        auto candidate = expand_set.at(i);
        if (!candidate.IsValid(param, num_leaves)) {
          continue;
        }
        this->ApplySplit(candidate, p_tree);

        num_leaves++;

        int left_child_nidx = tree[candidate.nid].LeftChild();
        int right_child_nidx = tree[candidate.nid].RightChild();
        // Only create child entries if needed_
        if (GPUExpandEntry::ChildIsValid(param, tree.GetDepth(left_child_nidx),
                                         num_leaves)) {
          monitor.Start("UpdatePosition");
          // Update position is only run when child is valid, instead of right after apply
          // split (as in approx tree method).  Hense we have the finalise position call
          // in GPU Hist.
          this->UpdatePosition(candidate, p_tree);
          monitor.Stop("UpdatePosition");

          monitor.Start("BuildHist");
          this->BuildHistLeftRight(candidate, left_child_nidx, right_child_nidx, reducer);
          monitor.Stop("BuildHist");

          monitor.Start("EvaluateSplits");
          this->EvaluateLeftRightSplits(candidate, left_child_nidx, right_child_nidx, *p_tree,
                                        new_candidates.subspan(i * 2, 2));
          monitor.Stop("EvaluateSplits");
        } else {
          // Set default
          new_candidates[i * 2] = GPUExpandEntry();
          new_candidates[i * 2 + 1] = GPUExpandEntry();
        }
      }
      dh::DefaultStream().Sync();
      driver.Push(new_candidates.begin(), new_candidates.end());
      expand_set = driver.Pop();
    }

    monitor.Start("FinalisePosition");
    this->FinalisePosition(p_tree, p_fmat, task, p_out_position);
    monitor.Stop("FinalisePosition");
  }
};

class GPUHistMaker : public TreeUpdater {
  using GradientSumT = GradientPairPrecise;

 public:
  explicit GPUHistMaker(GenericParameter const* ctx, ObjInfo task)
      : TreeUpdater(ctx), task_{task} {};
  void Configure(const Args& args) override {
    // Used in test to count how many configurations are performed
    LOG(DEBUG) << "[GPU Hist]: Configure";
    param_.UpdateAllowUnknown(args);
    hist_maker_param_.UpdateAllowUnknown(args);
    dh::CheckComputeCapability();
    initialised_ = false;

    monitor_.Init("updater_gpu_hist");
  }

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("gpu_hist_train_param"), &this->hist_maker_param_);
    initialised_ = false;
    FromJson(config.at("train_param"), &param_);
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["gpu_hist_train_param"] = ToJson(hist_maker_param_);
    out["train_param"] = ToJson(param_);
  }

  ~GPUHistMaker() {  // NOLINT
    dh::GlobalMemoryLogger().Log();
  }

  void Update(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat,
              common::Span<HostDeviceVector<bst_node_t>> out_position,
              const std::vector<RegTree*>& trees) override {
    monitor_.Start("Update");

    // rescale learning rate according to size of trees
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();

    // build tree
    try {
      size_t t_idx{0};
      for (xgboost::RegTree* tree : trees) {
        this->UpdateTree(gpair, dmat, tree, &out_position[t_idx]);

        if (hist_maker_param_.debug_synchronize) {
          this->CheckTreesSynchronized(tree);
        }
        ++t_idx;
      }
      dh::safe_cuda(cudaGetLastError());
    } catch (const std::exception& e) {
      LOG(FATAL) << "Exception in gpu_hist: " << e.what() << std::endl;
    }

    param_.learning_rate = lr;
    monitor_.Stop("Update");
  }

  void InitDataOnce(DMatrix* dmat) {
    CHECK_GE(ctx_->gpu_id, 0) << "Must have at least one device";
    info_ = &dmat->Info();
    reducer_.Init({ctx_->gpu_id});  // NOLINT

    // Synchronise the column sampling seed
    uint32_t column_sampling_seed = common::GlobalRandom()();
    rabit::Broadcast(&column_sampling_seed, sizeof(column_sampling_seed), 0);

    BatchParam batch_param{
      ctx_->gpu_id,
      param_.max_bin,
    };
    auto page = (*dmat->GetBatches<EllpackPage>(batch_param).begin()).Impl();
    dh::safe_cuda(cudaSetDevice(ctx_->gpu_id));
    info_->feature_types.SetDevice(ctx_->gpu_id);
    maker.reset(new GPUHistMakerDevice<GradientSumT>(
        ctx_, page, info_->feature_types.ConstDeviceSpan(), info_->num_row_, param_,
        column_sampling_seed, info_->num_col_, batch_param));

    p_last_fmat_ = dmat;
    initialised_ = true;
  }

  void InitData(DMatrix* dmat, RegTree const* p_tree) {
    if (!initialised_) {
      monitor_.Start("InitDataOnce");
      this->InitDataOnce(dmat);
      monitor_.Stop("InitDataOnce");
    }
    p_last_tree_ = p_tree;
  }

  // Only call this method for testing
  void CheckTreesSynchronized(RegTree* local_tree) const {
    std::string s_model;
    common::MemoryBufferStream fs(&s_model);
    int rank = rabit::GetRank();
    if (rank == 0) {
      local_tree->Save(&fs);
    }
    fs.Seek(0);
    rabit::Broadcast(&s_model, 0);
    RegTree reference_tree{};  // rank 0 tree
    reference_tree.Load(&fs);
    CHECK(*local_tree == reference_tree);
  }

  void UpdateTree(HostDeviceVector<GradientPair>* gpair, DMatrix* p_fmat, RegTree* p_tree,
                  HostDeviceVector<bst_node_t>* p_out_position) {
    monitor_.Start("InitData");
    this->InitData(p_fmat, p_tree);
    monitor_.Stop("InitData");

    gpair->SetDevice(ctx_->gpu_id);
    maker->UpdateTree(gpair, p_fmat, task_, p_tree, &reducer_, p_out_position);
  }

  bool UpdatePredictionCache(const DMatrix* data,
                             linalg::VectorView<bst_float> p_out_preds) override {
    if (maker == nullptr || p_last_fmat_ == nullptr || p_last_fmat_ != data) {
      return false;
    }
    monitor_.Start("UpdatePredictionCache");
    maker->UpdatePredictionCache(p_out_preds, p_last_tree_);
    monitor_.Stop("UpdatePredictionCache");
    return true;
  }

  TrainParam param_;  // NOLINT
  MetaInfo* info_{};  // NOLINT

  std::unique_ptr<GPUHistMakerDevice<GradientSumT>> maker;  // NOLINT

  char const* Name() const override { return "grow_gpu_hist"; }

 private:
  bool initialised_{false};

  GPUHistMakerTrainParam hist_maker_param_;

  dh::AllReducer reducer_;

  DMatrix* p_last_fmat_{nullptr};
  RegTree const* p_last_tree_{nullptr};
  ObjInfo task_;

  common::Monitor monitor_;
};

#if !defined(GTEST_TEST)
XGBOOST_REGISTER_TREE_UPDATER(GPUHistMaker, "grow_gpu_hist")
    .describe("Grow tree with GPU.")
    .set_body([](GenericParameter const* tparam, ObjInfo task) {
      return new GPUHistMaker(tparam, task);
    });
#endif  // !defined(GTEST_TEST)

}  // namespace tree
}  // namespace xgboost
