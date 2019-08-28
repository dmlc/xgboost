/*!
 * Copyright 2018 by Contributors
 * \author Rory Mitchell
 */

#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>
#include <xgboost/data.h>
#include <xgboost/linear_updater.h>
#include "../common/common.h"
#include "../common/span.h"
#include "../common/device_helpers.cuh"
#include "../common/timer.h"
#include "./param.h"
#include "coordinate_common.h"

namespace xgboost {
namespace linear {

DMLC_REGISTRY_FILE_TAG(updater_gpu_coordinate);

class DeviceShard {
  int device_id_;
  dh::BulkAllocator ba_;
  std::vector<size_t> row_ptr_;
  common::Span<xgboost::Entry> data_;
  common::Span<GradientPair> gpair_;
  dh::CubMemory temp_;
  size_t shard_size_;

 public:
  DeviceShard(int device_id,
              const SparsePage &batch,  // column batch
              bst_uint shard_size,
              const LinearTrainParam &param,
              const gbm::GBLinearModelParam &model_param)
      : device_id_(device_id),
        shard_size_(shard_size) {
    if ( IsEmpty() ) { return; }
    dh::safe_cuda(cudaSetDevice(device_id_));
    // The begin and end indices for the section of each column associated with
    // this shard
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
                           xgboost::Entry(shard_size_, 0.0f), cmp);
      column_segments.emplace_back(
          std::make_pair(column_begin - col.cbegin(), column_end - col.cbegin()));
      row_ptr_.push_back(row_ptr_.back() + (column_end - column_begin));
    }
    ba_.Allocate(device_id_, &data_, row_ptr_.back(), &gpair_,
                 shard_size_ * model_param.num_output_group);

    for (size_t fidx = 0; fidx < batch.Size(); fidx++) {
      auto col = batch[fidx];
      auto seg = column_segments[fidx];
      dh::safe_cuda(cudaMemcpy(
          data_.subspan(row_ptr_[fidx]).data(),
          col.data() + seg.first,
          sizeof(Entry) * (seg.second - seg.first), cudaMemcpyHostToDevice));
    }
  }

  ~DeviceShard() {  // NOLINT
    dh::safe_cuda(cudaSetDevice(device_id_));
  }

  bool IsEmpty() {
    return shard_size_ == 0;
  }

  void UpdateGpair(const std::vector<GradientPair> &host_gpair,
                   const gbm::GBLinearModelParam &model_param) {
    dh::safe_cuda(cudaMemcpyAsync(
        gpair_.data(),
        host_gpair.data(),
        gpair_.size() * sizeof(GradientPair), cudaMemcpyHostToDevice));
  }

  GradientPair GetBiasGradient(int group_idx, int num_group) {
    dh::safe_cuda(cudaSetDevice(device_id_));
    auto counting = thrust::make_counting_iterator(0ull);
    auto f = [=] __device__(size_t idx) {
      return idx * num_group + group_idx;
    };  // NOLINT
    thrust::transform_iterator<decltype(f), decltype(counting), size_t> skip(
        counting, f);
    auto perm = thrust::make_permutation_iterator(gpair_.data(), skip);

    return dh::SumReduction(temp_, perm, shard_size_);
  }

  void UpdateBiasResidual(float dbias, int group_idx, int num_groups) {
    if (dbias == 0.0f) return;
    auto d_gpair = gpair_;
    dh::LaunchN(device_id_, shard_size_, [=] __device__(size_t idx) {
      auto &g = d_gpair[idx * num_groups + group_idx];
      g += GradientPair(g.GetHess() * dbias, 0);
    });
  }

  GradientPair GetGradient(int group_idx, int num_group, int fidx) {
    dh::safe_cuda(cudaSetDevice(device_id_));
    common::Span<xgboost::Entry> d_col = data_.subspan(row_ptr_[fidx]);
    size_t col_size = row_ptr_[fidx + 1] - row_ptr_[fidx];
    common::Span<GradientPair> d_gpair = gpair_;
    auto counting = thrust::make_counting_iterator(0ull);
    auto f = [=] __device__(size_t idx) {
      auto entry = d_col[idx];
      auto g = d_gpair[entry.index * num_group + group_idx];
      return GradientPair(g.GetGrad() * entry.fvalue,
                          g.GetHess() * entry.fvalue * entry.fvalue);
    };  // NOLINT
    thrust::transform_iterator<decltype(f), decltype(counting), GradientPair>
        multiply_iterator(counting, f);
    return dh::SumReduction(temp_, multiply_iterator, col_size);
  }

  void UpdateResidual(float dw, int group_idx, int num_groups, int fidx) {
    common::Span<GradientPair> d_gpair = gpair_;
    common::Span<Entry> d_col = data_.subspan(row_ptr_[fidx]);
    size_t col_size = row_ptr_[fidx + 1] - row_ptr_[fidx];
    dh::LaunchN(device_id_, col_size, [=] __device__(size_t idx) {
      auto entry = d_col[idx];
      auto &g = d_gpair[entry.index * num_groups + group_idx];
      g += GradientPair(g.GetHess() * dw * entry.fvalue, 0);
    });
  }
};

/**
 * \class GPUCoordinateUpdater
 *
 * \brief Coordinate descent algorithm that updates one feature per iteration
 */

class GPUCoordinateUpdater : public LinearUpdater {  // NOLINT
 public:
  // set training parameter
  void Configure(Args const& args) override {
    tparam_.InitAllowUnknown(args);
    selector_.reset(FeatureSelector::Create(tparam_.feature_selector));
    monitor_.Init("GPUCoordinateUpdater");
  }

  void LazyInitShards(DMatrix *p_fmat,
                      const gbm::GBLinearModelParam &model_param) {
    if (shard_) return;

    device_ = learner_param_->gpu_id;

    auto num_row = static_cast<size_t>(p_fmat->Info().num_row_);

    // Partition input matrix into row segments
    std::vector<size_t> row_segments;
    row_segments.push_back(0);
    size_t shard_size = num_row;
    row_segments.push_back(shard_size);

    CHECK(p_fmat->SingleColBlock());
    SparsePage const& batch = *(p_fmat->GetBatches<CSCPage>().begin());

    // Create device shard
    shard_.reset(new DeviceShard(device_, batch, shard_size, tparam_, model_param));
  }

  void Update(HostDeviceVector<GradientPair> *in_gpair, DMatrix *p_fmat,
              gbm::GBLinearModel *model, double sum_instance_weight) override {
    tparam_.DenormalizePenalties(sum_instance_weight);
    monitor_.Start("LazyInitShards");
    this->LazyInitShards(p_fmat, model->param);
    monitor_.Stop("LazyInitShards");

    monitor_.Start("UpdateGpair");
    auto &in_gpair_host = in_gpair->ConstHostVector();
    // Update gpair
    if (shard_) {
      shard_->UpdateGpair(in_gpair_host, model->param);
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
    for (auto group_idx = 0; group_idx < model->param.num_output_group;
         ++group_idx) {
      for (auto i = 0U; i < model->param.num_feature; i++) {
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
    for (int group_idx = 0; group_idx < model->param.num_output_group; ++group_idx) {
      // Get gradient
      auto grad = GradientPair(0, 0);
      if (shard_) {
        grad = shard_->GetBiasGradient(group_idx, model->param.num_output_group);
      }
      auto dbias = static_cast<float>(
          tparam_.learning_rate *
              CoordinateDeltaBias(grad.GetGrad(), grad.GetHess()));
      model->bias()[group_idx] += dbias;

      // Update residual
      if (shard_) {
        shard_->UpdateBiasResidual(dbias, group_idx, model->param.num_output_group);
      }
    }
  }

  void UpdateFeature(int fidx, int group_idx,
                     std::vector<GradientPair> *in_gpair,
                     gbm::GBLinearModel *model) {
    bst_float &w = (*model)[fidx][group_idx];
    // Get gradient
    auto grad = GradientPair(0, 0);
    if (shard_) {
      grad = shard_->GetGradient(group_idx, model->param.num_output_group, fidx);
    }
    auto dw = static_cast<float>(tparam_.learning_rate *
                                 CoordinateDelta(grad.GetGrad(), grad.GetHess(),
                                                 w, tparam_.reg_alpha_denorm,
                                                 tparam_.reg_lambda_denorm));
    w += dw;

    if (shard_) {
      shard_->UpdateResidual(dw, group_idx, model->param.num_output_group, fidx);
    }
  }

 private:
  // training parameter
  LinearTrainParam tparam_;
  CoordinateParam coord_param_;
  int device_{};
  std::unique_ptr<FeatureSelector> selector_;
  common::Monitor monitor_;

  std::unique_ptr<DeviceShard> shard_{nullptr};
};

XGBOOST_REGISTER_LINEAR_UPDATER(GPUCoordinateUpdater, "gpu_coord_descent")
    .describe(
        "Update linear model according to coordinate descent algorithm. GPU "
        "accelerated.")
    .set_body([]() { return new GPUCoordinateUpdater(); });
}  // namespace linear
}  // namespace xgboost
