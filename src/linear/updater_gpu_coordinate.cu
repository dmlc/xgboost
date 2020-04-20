/*!
 * Copyright 2018-2019 by Contributors
 * \author Rory Mitchell
 */

#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>
#include <xgboost/data.h>
#include <xgboost/linear_updater.h>
#include "xgboost/span.h"

#include "coordinate_common.h"
#include "../common/common.h"
#include "../common/device_helpers.cuh"
#include "../common/timer.h"
#include "./param.h"

namespace xgboost {
namespace linear {

DMLC_REGISTRY_FILE_TAG(updater_gpu_coordinate);

/**
 * \class GPUCoordinateUpdater
 *
 * \brief Coordinate descent algorithm that updates one feature per iteration
 */

class GPUCoordinateUpdater : public LinearUpdater {  // NOLINT
 public:
  ~GPUCoordinateUpdater() {  // NOLINT
    if (learner_param_->gpu_id >= 0) {
      dh::safe_cuda(cudaSetDevice(learner_param_->gpu_id));
    }
  }

  // set training parameter
  void Configure(Args const& args) override {
    tparam_.UpdateAllowUnknown(args);
    coord_param_.UpdateAllowUnknown(args);
    selector_.reset(FeatureSelector::Create(tparam_.feature_selector));
    monitor_.Init("GPUCoordinateUpdater");
  }

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("linear_train_param"), &tparam_);
    FromJson(config.at("coordinate_param"), &coord_param_);
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["linear_train_param"] = ToJson(tparam_);
    out["coordinate_param"] = ToJson(coord_param_);
  }

  void LazyInitDevice(DMatrix *p_fmat, const LearnerModelParam &model_param) {
    if (learner_param_->gpu_id < 0) return;

    num_row_ = static_cast<size_t>(p_fmat->Info().num_row_);

    CHECK(p_fmat->SingleColBlock());
    SparsePage const& batch = *(p_fmat->GetBatches<CSCPage>().begin());

    if (IsEmpty()) {
      return;
    }

    dh::safe_cuda(cudaSetDevice(learner_param_->gpu_id));
    // The begin and end indices for the section of each column associated with
    // this device
    std::vector<std::pair<bst_uint, bst_uint>> column_segments;
    row_ptr_ = {0};
    // iterate through columns
    for (size_t fidx = 0; fidx < batch.Size(); fidx++) {
      common::Span<Entry const> col = batch[fidx];
      auto cmp = [](Entry e1, Entry e2) {
        return e1.index < e2.index;
      };
      auto column_begin =
          std::lower_bound(col.cbegin(), col.cend(),
                           xgboost::Entry(0, 0.0f), cmp);
      auto column_end =
          std::lower_bound(col.cbegin(), col.cend(),
                           xgboost::Entry(num_row_, 0.0f), cmp);
      column_segments.emplace_back(
          std::make_pair(column_begin - col.cbegin(), column_end - col.cbegin()));
      row_ptr_.push_back(row_ptr_.back() + (column_end - column_begin));
    }
    data_.resize(row_ptr_.back());
    gpair_.resize(num_row_ * model_param.num_output_group);
    for (size_t fidx = 0; fidx < batch.Size(); fidx++) {
      auto col = batch[fidx];
      auto seg = column_segments[fidx];
      dh::safe_cuda(cudaMemcpy(
          data_.data().get() + row_ptr_[fidx],
          col.data() + seg.first,
          sizeof(Entry) * (seg.second - seg.first), cudaMemcpyHostToDevice));
    }
  }

  void Update(HostDeviceVector<GradientPair> *in_gpair, DMatrix *p_fmat,
              gbm::GBLinearModel *model, double sum_instance_weight) override {
    tparam_.DenormalizePenalties(sum_instance_weight);
    monitor_.Start("LazyInitDevice");
    this->LazyInitDevice(p_fmat, *(model->learner_model_param));
    monitor_.Stop("LazyInitDevice");

    monitor_.Start("UpdateGpair");
    auto &in_gpair_host = in_gpair->ConstHostVector();
    // Update gpair
    if (learner_param_->gpu_id >= 0) {
      this->UpdateGpair(in_gpair_host);
    }
    monitor_.Stop("UpdateGpair");

    monitor_.Start("UpdateBias");
    this->UpdateBias(p_fmat, model);
    monitor_.Stop("UpdateBias");
    // prepare for updating the weights
    selector_->Setup(*model, in_gpair->ConstHostVector(), p_fmat,
                     tparam_.reg_alpha_denorm, tparam_.reg_lambda_denorm,
                     coord_param_.top_k);
    monitor_.Start("UpdateFeature");
    for (auto group_idx = 0; group_idx < model->learner_model_param->num_output_group;
         ++group_idx) {
      for (auto i = 0U; i < model->learner_model_param->num_feature; i++) {
        auto fidx = selector_->NextFeature(
            i, *model, group_idx, in_gpair->ConstHostVector(), p_fmat,
            tparam_.reg_alpha_denorm, tparam_.reg_lambda_denorm);
        if (fidx < 0) break;
        this->UpdateFeature(fidx, group_idx, &in_gpair->HostVector(), model);
      }
    }
    monitor_.Stop("UpdateFeature");
  }

  void UpdateBias(DMatrix *p_fmat, gbm::GBLinearModel *model) {
    for (int group_idx = 0; group_idx < model->learner_model_param->num_output_group;
         ++group_idx) {
      // Get gradient
      auto grad = GradientPair(0, 0);
      if (learner_param_->gpu_id >= 0) {
        grad = GetBiasGradient(group_idx, model->learner_model_param->num_output_group);
      }
      auto dbias = static_cast<float>(
          tparam_.learning_rate *
              CoordinateDeltaBias(grad.GetGrad(), grad.GetHess()));
      model->Bias()[group_idx] += dbias;

      // Update residual
      if (learner_param_->gpu_id >= 0) {
        UpdateBiasResidual(dbias, group_idx, model->learner_model_param->num_output_group);
      }
    }
  }

  void UpdateFeature(int fidx, int group_idx,
                     std::vector<GradientPair> *in_gpair,
                     gbm::GBLinearModel *model) {
    bst_float &w = (*model)[fidx][group_idx];
    // Get gradient
    auto grad = GradientPair(0, 0);
    if (learner_param_->gpu_id >= 0) {
      grad = GetGradient(group_idx, model->learner_model_param->num_output_group, fidx);
    }
    auto dw = static_cast<float>(tparam_.learning_rate *
                                 CoordinateDelta(grad.GetGrad(), grad.GetHess(),
                                                 w, tparam_.reg_alpha_denorm,
                                                 tparam_.reg_lambda_denorm));
    w += dw;

    if (learner_param_->gpu_id >= 0) {
      UpdateResidual(dw, group_idx, model->learner_model_param->num_output_group, fidx);
    }
  }

  // This needs to be public because of the __device__ lambda.
  GradientPair GetBiasGradient(int group_idx, int num_group) {
    dh::safe_cuda(cudaSetDevice(learner_param_->gpu_id));
    auto counting = thrust::make_counting_iterator(0ull);
    auto f = [=] __device__(size_t idx) {
      return idx * num_group + group_idx;
    };  // NOLINT
    thrust::transform_iterator<decltype(f), decltype(counting), size_t> skip(
        counting, f);
    auto perm = thrust::make_permutation_iterator(gpair_.data(), skip);

    return dh::SumReduction(perm, num_row_);
  }

  // This needs to be public because of the __device__ lambda.
  void UpdateBiasResidual(float dbias, int group_idx, int num_groups) {
    if (dbias == 0.0f) return;
    auto d_gpair = dh::ToSpan(gpair_);
    dh::LaunchN(learner_param_->gpu_id, num_row_, [=] __device__(size_t idx) {
      auto &g = d_gpair[idx * num_groups + group_idx];
      g += GradientPair(g.GetHess() * dbias, 0);
    });
  }

  // This needs to be public because of the __device__ lambda.
  GradientPair GetGradient(int group_idx, int num_group, int fidx) {
    dh::safe_cuda(cudaSetDevice(learner_param_->gpu_id));
    common::Span<xgboost::Entry> d_col = dh::ToSpan(data_).subspan(row_ptr_[fidx]);
    size_t col_size = row_ptr_[fidx + 1] - row_ptr_[fidx];
    common::Span<GradientPair> d_gpair = dh::ToSpan(gpair_);
    auto counting = thrust::make_counting_iterator(0ull);
    auto f = [=] __device__(size_t idx) {
      auto entry = d_col[idx];
      auto g = d_gpair[entry.index * num_group + group_idx];
      return GradientPair(g.GetGrad() * entry.fvalue,
                          g.GetHess() * entry.fvalue * entry.fvalue);
    };  // NOLINT
    thrust::transform_iterator<decltype(f), decltype(counting), GradientPair>
        multiply_iterator(counting, f);
    return dh::SumReduction(multiply_iterator, col_size);
  }

  // This needs to be public because of the __device__ lambda.
  void UpdateResidual(float dw, int group_idx, int num_groups, int fidx) {
    common::Span<GradientPair> d_gpair = dh::ToSpan(gpair_);
    common::Span<Entry> d_col = dh::ToSpan(data_).subspan(row_ptr_[fidx]);
    size_t col_size = row_ptr_[fidx + 1] - row_ptr_[fidx];
    dh::LaunchN(learner_param_->gpu_id, col_size, [=] __device__(size_t idx) {
      auto entry = d_col[idx];
      auto &g = d_gpair[entry.index * num_groups + group_idx];
      g += GradientPair(g.GetHess() * dw * entry.fvalue, 0);
    });
  }

 private:
  bool IsEmpty() {
    return num_row_ == 0;
  }

  void UpdateGpair(const std::vector<GradientPair> &host_gpair) {
    dh::safe_cuda(cudaMemcpyAsync(
        gpair_.data().get(),
        host_gpair.data(),
        gpair_.size() * sizeof(GradientPair), cudaMemcpyHostToDevice));
  }

  // training parameter
  LinearTrainParam tparam_;
  CoordinateParam coord_param_;
  std::unique_ptr<FeatureSelector> selector_;
  common::Monitor monitor_;

  std::vector<size_t> row_ptr_;
  dh::device_vector<xgboost::Entry> data_;
  dh::caching_device_vector<GradientPair> gpair_;
  size_t num_row_;
};

XGBOOST_REGISTER_LINEAR_UPDATER(GPUCoordinateUpdater, "gpu_coord_descent")
    .describe(
        "Update linear model according to coordinate descent algorithm. GPU "
        "accelerated.")
    .set_body([]() { return new GPUCoordinateUpdater(); });
}  // namespace linear
}  // namespace xgboost
