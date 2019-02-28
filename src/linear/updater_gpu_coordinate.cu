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

void RescaleIndices(int device_idx, size_t ridx_begin,
                    common::Span<xgboost::Entry> data) {
  dh::LaunchN(device_idx, data.size(),
              [=] __device__(size_t idx) { data[idx].index -= ridx_begin; });
}

class DeviceShard {
  int device_id_;
  dh::BulkAllocator ba_;
  std::vector<size_t> row_ptr_;
  common::Span<xgboost::Entry> data_;
  common::Span<GradientPair> gpair_;
  dh::CubMemory temp_;
  size_t ridx_begin_;
  size_t ridx_end_;

 public:
  DeviceShard(int device_id,
              const SparsePage &batch,  // column batch
              bst_uint row_begin, bst_uint row_end,
              const LinearTrainParam &param,
              const gbm::GBLinearModelParam &model_param)
      : device_id_(device_id), ridx_begin_(row_begin), ridx_end_(row_end) {
    if (IsEmpty()) {
      return;
    }
    dh::safe_cuda(cudaSetDevice(device_id_));
    // The begin and end indices for the section of each column associated with
    // this shard
    std::vector<std::pair<bst_uint, bst_uint>> column_segments;
    row_ptr_ = {0};
    // iterate through columns
    for (auto fidx = 0; fidx < batch.Size(); fidx++) {
      common::Span<Entry const> col = batch[fidx];
      auto cmp = [](Entry e1, Entry e2) { return e1.index < e2.index; };
      auto column_begin = std::lower_bound(
          col.cbegin(), col.cend(), xgboost::Entry(row_begin, 0.0f), cmp);
      auto column_end = std::lower_bound(col.cbegin(), col.cend(),
                                         xgboost::Entry(row_end, 0.0f), cmp);
      column_segments.emplace_back(std::make_pair(column_begin - col.cbegin(),
                                                  column_end - col.cbegin()));
      row_ptr_.push_back(row_ptr_.back() + (column_end - column_begin));
    }
    ba_.Allocate(device_id_, &data_, row_ptr_.back(), &gpair_,
                 (row_end - row_begin) * model_param.num_output_group);

    for (int fidx = 0; fidx < batch.Size(); fidx++) {
      auto col = batch[fidx];
      auto seg = column_segments[fidx];
      dh::safe_cuda(cudaMemcpy(
          data_.subspan(row_ptr_[fidx]).data(),
          col.data() + seg.first,
          sizeof(Entry) * (seg.second - seg.first), cudaMemcpyHostToDevice));
    }
    // Rescale indices with respect to current shard
    RescaleIndices(device_id_, ridx_begin_, data_);
  }

  bool IsEmpty() {
    return (ridx_end_ - ridx_begin_) == 0;
  }

  void UpdateGpair(const std::vector<GradientPair> &host_gpair,
                   const gbm::GBLinearModelParam &model_param) {
    dh::safe_cuda(cudaMemcpyAsync(
        gpair_.data(),
        host_gpair.data() + ridx_begin_ * model_param.num_output_group,
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

    return dh::SumReduction(temp_, perm, ridx_end_ - ridx_begin_);
  }

  void UpdateBiasResidual(float dbias, int group_idx, int num_groups) {
    if (dbias == 0.0f) return;
    auto d_gpair = gpair_;
    dh::LaunchN(device_id_, ridx_end_ - ridx_begin_, [=] __device__(size_t idx) {
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

class GPUCoordinateUpdater : public LinearUpdater {
 public:
  // set training parameter
  void Init(
      const std::vector<std::pair<std::string, std::string>> &args) override {
    tparam_.InitAllowUnknown(args);
    selector_.reset(FeatureSelector::Create(tparam_.feature_selector));
    monitor_.Init("GPUCoordinateUpdater");
  }

  void LazyInitShards(DMatrix *p_fmat,
                      const gbm::GBLinearModelParam &model_param) {
    if (!shards_.empty()) return;
    dist_ = GPUDistribution::Block(GPUSet::All(tparam_.gpu_id, tparam_.n_gpus,
                                               p_fmat->Info().num_row_));
    auto devices = dist_.Devices();

    size_t n_devices = static_cast<size_t>(devices.Size());
    size_t row_begin = 0;
    size_t num_row = static_cast<size_t>(p_fmat->Info().num_row_);

    // Partition input matrix into row segments
    std::vector<size_t> row_segments;
    row_segments.push_back(0);
    for (int d_idx = 0; d_idx < n_devices; ++d_idx) {
      size_t shard_size = dist_.ShardSize(num_row, d_idx);
      size_t row_end = row_begin + shard_size;
      row_segments.push_back(row_end);
      row_begin = row_end;
    }

    CHECK(p_fmat->SingleColBlock());
    SparsePage const& batch = *(p_fmat->GetColumnBatches().begin());

    shards_.resize(n_devices);
    // Create device shards
    dh::ExecuteIndexShards(&shards_,
                           [&](int i, std::unique_ptr<DeviceShard>& shard) {
        shard = std::unique_ptr<DeviceShard>(
            new DeviceShard(devices.DeviceId(i), batch, row_segments[i],
                            row_segments[i + 1], tparam_, model_param));
      });
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
    dh::ExecuteIndexShards(&shards_, [&](int idx, std::unique_ptr<DeviceShard>& shard) {
      if (!shard->IsEmpty()) {
        shard->UpdateGpair(in_gpair_host, model->param);
      }
    });
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
    for (int group_idx = 0; group_idx < model->param.num_output_group;
         ++group_idx) {
      // Get gradient
      auto grad = dh::ReduceShards<GradientPair>(
          &shards_, [&](std::unique_ptr<DeviceShard> &shard) {
            if (!shard->IsEmpty()) {
              GradientPair result =
                  shard->GetBiasGradient(group_idx,
                                         model->param.num_output_group);
              return result;
            }
            return GradientPair(0, 0);
          });

      auto dbias = static_cast<float>(
          tparam_.learning_rate *
          CoordinateDeltaBias(grad.GetGrad(), grad.GetHess()));
      model->bias()[group_idx] += dbias;

      // Update residual
    dh::ExecuteIndexShards(&shards_, [&](int idx, std::unique_ptr<DeviceShard>& shard) {
        if (!shard->IsEmpty()) {
          shard->UpdateBiasResidual(dbias, group_idx,
                                    model->param.num_output_group);
        }
      });
    }
  }

  void UpdateFeature(int fidx, int group_idx,
                     std::vector<GradientPair> *in_gpair,
                     gbm::GBLinearModel *model) {
    bst_float &w = (*model)[fidx][group_idx];
    // Get gradient
    auto grad = dh::ReduceShards<GradientPair>(
        &shards_, [&](std::unique_ptr<DeviceShard> &shard) {
          if (!shard->IsEmpty()) {
            return shard->GetGradient(group_idx, model->param.num_output_group,
                                      fidx);
          }
          return GradientPair(0, 0);
        });

    auto dw = static_cast<float>(tparam_.learning_rate *
                                 CoordinateDelta(grad.GetGrad(), grad.GetHess(),
                                                 w, tparam_.reg_alpha_denorm,
                                                 tparam_.reg_lambda_denorm));
    w += dw;

    dh::ExecuteIndexShards(&shards_, [&](int idx,
                                        std::unique_ptr<DeviceShard> &shard) {
      if (!shard->IsEmpty()) {
        shard->UpdateResidual(dw, group_idx, model->param.num_output_group, fidx);
      }
    });
  }

 private:
  // training parameter
  LinearTrainParam tparam_;
  CoordinateParam coord_param_;
  GPUDistribution dist_;
  std::unique_ptr<FeatureSelector> selector_;
  common::Monitor monitor_;

  std::vector<std::unique_ptr<DeviceShard>> shards_;
};

XGBOOST_REGISTER_LINEAR_UPDATER(GPUCoordinateUpdater, "gpu_coord_descent")
    .describe(
        "Update linear model according to coordinate descent algorithm. GPU "
        "accelerated.")
    .set_body([]() { return new GPUCoordinateUpdater(); });
}  // namespace linear
}  // namespace xgboost
