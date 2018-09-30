/*!
 * Copyright 2018 by Contributors
 * \author Rory Mitchell
 */

#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>
#include <xgboost/linear_updater.h>
#include "../common/common.h"
#include "../common/device_helpers.cuh"
#include "../common/timer.h"
#include "coordinate_common.h"

namespace xgboost {
namespace linear {

DMLC_REGISTRY_FILE_TAG(updater_gpu_coordinate);

// training parameter
struct GPUCoordinateTrainParam
    : public dmlc::Parameter<GPUCoordinateTrainParam> {
  /*! \brief learning_rate */
  float learning_rate;
  /*! \brief regularization weight for L2 norm */
  float reg_lambda;
  /*! \brief regularization weight for L1 norm */
  float reg_alpha;
  int feature_selector;
  int top_k;
  int debug_verbose;
  int n_gpus;
  int gpu_id;
  bool silent;
  // declare parameters
  DMLC_DECLARE_PARAMETER(GPUCoordinateTrainParam) {
    DMLC_DECLARE_FIELD(learning_rate)
        .set_lower_bound(0.0f)
        .set_default(1.0f)
        .describe("Learning rate of each update.");
    DMLC_DECLARE_FIELD(reg_lambda)
        .set_lower_bound(0.0f)
        .set_default(0.0f)
        .describe("L2 regularization on weights.");
    DMLC_DECLARE_FIELD(reg_alpha)
        .set_lower_bound(0.0f)
        .set_default(0.0f)
        .describe("L1 regularization on weights.");
    DMLC_DECLARE_FIELD(feature_selector)
        .set_default(kCyclic)
        .add_enum("cyclic", kCyclic)
        .add_enum("shuffle", kShuffle)
        .add_enum("thrifty", kThrifty)
        .add_enum("greedy", kGreedy)
        .add_enum("random", kRandom)
        .describe("Feature selection or ordering method.");
    DMLC_DECLARE_FIELD(top_k).set_lower_bound(0).set_default(0).describe(
        "The number of top features to select in 'thrifty' feature_selector. "
        "The value of zero means using all the features.");
    DMLC_DECLARE_FIELD(debug_verbose)
        .set_lower_bound(0)
        .set_default(0)
        .describe("flag to print out detailed breakdown of runtime");
    DMLC_DECLARE_FIELD(n_gpus).set_default(1).describe(
        "Number of devices to use.");
    DMLC_DECLARE_FIELD(gpu_id).set_default(0).describe(
        "Primary device ordinal.");
    DMLC_DECLARE_FIELD(silent).set_default(false).describe(
        "Do not print information during trainig.");
    // alias of parameters
    DMLC_DECLARE_ALIAS(learning_rate, eta);
    DMLC_DECLARE_ALIAS(reg_lambda, lambda);
    DMLC_DECLARE_ALIAS(reg_alpha, alpha);
  }
  /*! \brief Denormalizes the regularization penalties - to be called at each
   * update */
  void DenormalizePenalties(double sum_instance_weight) {
    reg_lambda_denorm = reg_lambda * sum_instance_weight;
    reg_alpha_denorm = reg_alpha * sum_instance_weight;
  }
  // denormalizated regularization penalties
  float reg_lambda_denorm;
  float reg_alpha_denorm;
};

void RescaleIndices(size_t ridx_begin, dh::DVec<Entry> *data) {
  auto d_data = data->Data();
  dh::LaunchN(data->DeviceIdx(), data->Size(),
              [=] __device__(size_t idx) { d_data[idx].index -= ridx_begin; });
}

class DeviceShard {
  int device_idx_;
  int normalised_device_idx_;  // Device index counting from param.gpu_id
  dh::BulkAllocator<dh::MemoryType::kDevice> ba_;
  std::vector<size_t> row_ptr_;
  dh::DVec<Entry> data_;
  dh::DVec<GradientPair> gpair_;
  dh::CubMemory temp_;
  size_t ridx_begin_;
  size_t ridx_end_;

 public:
  DeviceShard(int device_idx, int normalised_device_idx, const SparsePage &batch,
              bst_uint row_begin, bst_uint row_end,
              const GPUCoordinateTrainParam &param,
              const gbm::GBLinearModelParam &model_param)
      : device_idx_(device_idx),
        normalised_device_idx_(normalised_device_idx),
        ridx_begin_(row_begin),
        ridx_end_(row_end) {
    dh::safe_cuda(cudaSetDevice(device_idx));
    // The begin and end indices for the section of each column associated with
    // this shard
    std::vector<std::pair<bst_uint, bst_uint>> column_segments;
    row_ptr_ = {0};
    for (auto fidx = 0; fidx < batch.Size(); fidx++) {
      auto col = batch[fidx];
      auto cmp = [](Entry e1, Entry e2) {
        return e1.index < e2.index;
      };
      auto column_begin =
          std::lower_bound(col.data(), col.data() + col.size(),
                           Entry(row_begin, 0.0f), cmp);
      auto column_end =
          std::upper_bound(col.data(), col.data() + col.size(),
                           Entry(row_end, 0.0f), cmp);
      column_segments.push_back(
          std::make_pair(column_begin - col.data(), column_end - col.data()));
      row_ptr_.push_back(row_ptr_.back() + column_end - column_begin);
    }
    ba_.Allocate(device_idx, param.silent, &data_, row_ptr_.back(), &gpair_,
                (row_end - row_begin) * model_param.num_output_group);

    for (int fidx = 0; fidx < batch.Size(); fidx++) {
      auto col = batch[fidx];
      auto seg = column_segments[fidx];
      dh::safe_cuda(cudaMemcpy(
          data_.Data() + row_ptr_[fidx], col.data() + seg.first,
          sizeof(Entry) * (seg.second - seg.first), cudaMemcpyHostToDevice));
    }
    // Rescale indices with respect to current shard
    RescaleIndices(ridx_begin_, &data_);
  }
  void UpdateGpair(const std::vector<GradientPair> &host_gpair,
                   const gbm::GBLinearModelParam &model_param) {
    gpair_.copy(host_gpair.begin() + ridx_begin_ * model_param.num_output_group,
               host_gpair.begin() + ridx_end_ * model_param.num_output_group);
  }

  GradientPair GetBiasGradient(int group_idx, int num_group) {
    auto counting = thrust::make_counting_iterator(0ull);
    auto f = [=] __device__(size_t idx) {
      return idx * num_group + group_idx;
    };  // NOLINT
    thrust::transform_iterator<decltype(f), decltype(counting), size_t> skip(
        counting, f);
    auto perm = thrust::make_permutation_iterator(gpair_.tbegin(), skip);

    return dh::SumReduction(temp_, perm, ridx_end_ - ridx_begin_);
  }

  void UpdateBiasResidual(float dbias, int group_idx, int num_groups) {
    if (dbias == 0.0f) return;
    auto d_gpair = gpair_.Data();
    dh::LaunchN(device_idx_, ridx_end_ - ridx_begin_, [=] __device__(size_t idx) {
      auto &g = d_gpair[idx * num_groups + group_idx];
      g += GradientPair(g.GetHess() * dbias, 0);
    });
  }

  GradientPair GetGradient(int group_idx, int num_group, int fidx) {
    auto d_col = data_.Data() + row_ptr_[fidx];
    size_t col_size = row_ptr_[fidx + 1] - row_ptr_[fidx];
    auto d_gpair = gpair_.Data();
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
    auto d_gpair = gpair_.Data();
    auto d_col = data_.Data() + row_ptr_[fidx];
    size_t col_size = row_ptr_[fidx + 1] - row_ptr_[fidx];
    dh::LaunchN(device_idx_, col_size, [=] __device__(size_t idx) {
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
    param.InitAllowUnknown(args);
    selector.reset(FeatureSelector::Create(param.feature_selector));
    monitor.Init("GPUCoordinateUpdater", param.debug_verbose);
  }

  void LazyInitShards(DMatrix *p_fmat,
                      const gbm::GBLinearModelParam &model_param) {
    if (!shards.empty()) return;
    int n_devices = GPUSet::All(param.n_gpus, p_fmat->Info().num_row_).Size();
    bst_uint row_begin = 0;
    bst_uint shard_size =
        std::ceil(static_cast<double>(p_fmat->Info().num_row_) / n_devices);

    device_list.resize(n_devices);
    for (int d_idx = 0; d_idx < n_devices; ++d_idx) {
      int device_idx = GPUSet::GetDeviceIdx(param.gpu_id + d_idx);
      device_list[d_idx] = device_idx;
    }
    // Partition input matrix into row segments
    std::vector<size_t> row_segments;
    row_segments.push_back(0);
    for (int d_idx = 0; d_idx < n_devices; ++d_idx) {
      bst_uint row_end = std::min(static_cast<size_t>(row_begin + shard_size),
                                  p_fmat->Info().num_row_);
      row_segments.push_back(row_end);
      row_begin = row_end;
    }

    CHECK(p_fmat->SingleColBlock());
    const auto &batch = *p_fmat->GetColumnBatches().begin();

    shards.resize(n_devices);
    // Create device shards
    dh::ExecuteShards(&shards, [&](std::unique_ptr<DeviceShard> &shard) {
      auto idx = &shard - &shards[0];
      shard = std::unique_ptr<DeviceShard>(
          new DeviceShard(device_list[idx], idx, batch, row_segments[idx],
                          row_segments[idx + 1], param, model_param));
    });
  }
  void Update(HostDeviceVector<GradientPair> *in_gpair, DMatrix *p_fmat,
              gbm::GBLinearModel *model, double sum_instance_weight) override {
    param.DenormalizePenalties(sum_instance_weight);
    monitor.Start("LazyInitShards");
    this->LazyInitShards(p_fmat, model->param);
    monitor.Stop("LazyInitShards");

    monitor.Start("UpdateGpair");
    // Update gpair
    dh::ExecuteShards(&shards, [&](std::unique_ptr<DeviceShard> &shard) {
      shard->UpdateGpair(in_gpair->ConstHostVector(), model->param);
    });
    monitor.Stop("UpdateGpair");

    monitor.Start("UpdateBias");
    this->UpdateBias(p_fmat, model);
    monitor.Stop("UpdateBias");
    // prepare for updating the weights
    selector->Setup(*model, in_gpair->ConstHostVector(), p_fmat,
                    param.reg_alpha_denorm, param.reg_lambda_denorm,
                    param.top_k);
    monitor.Start("UpdateFeature");
    for (auto group_idx = 0; group_idx < model->param.num_output_group;
         ++group_idx) {
      for (auto i = 0U; i < model->param.num_feature; i++) {
        auto fidx = selector->NextFeature(
            i, *model, group_idx, in_gpair->ConstHostVector(), p_fmat,
            param.reg_alpha_denorm, param.reg_lambda_denorm);
        if (fidx < 0) break;
        this->UpdateFeature(fidx, group_idx, &in_gpair->HostVector(), model);
      }
    }
    monitor.Stop("UpdateFeature");
  }

  void UpdateBias(DMatrix *p_fmat, gbm::GBLinearModel *model) {
    for (int group_idx = 0; group_idx < model->param.num_output_group;
         ++group_idx) {
      // Get gradient
      auto grad = dh::ReduceShards<GradientPair>(
          &shards, [&](std::unique_ptr<DeviceShard> &shard) {
            return shard->GetBiasGradient(group_idx,
                                          model->param.num_output_group);
          });

      auto dbias = static_cast<float>(
          param.learning_rate *
          CoordinateDeltaBias(grad.GetGrad(), grad.GetHess()));
      model->bias()[group_idx] += dbias;

      // Update residual
      dh::ExecuteShards(&shards, [&](std::unique_ptr<DeviceShard> &shard) {
        shard->UpdateBiasResidual(dbias, group_idx,
                                  model->param.num_output_group);
      });
    }
  }

  void UpdateFeature(int fidx, int group_idx,
                     std::vector<GradientPair> *in_gpair,
                     gbm::GBLinearModel *model) {
    bst_float &w = (*model)[fidx][group_idx];
    // Get gradient
    auto grad = dh::ReduceShards<GradientPair>(
        &shards, [&](std::unique_ptr<DeviceShard> &shard) {
          return shard->GetGradient(group_idx, model->param.num_output_group,
                                    fidx);
        });

    auto dw = static_cast<float>(param.learning_rate *
                                 CoordinateDelta(grad.GetGrad(), grad.GetHess(),
                                                 w, param.reg_alpha_denorm,
                                                 param.reg_lambda_denorm));
    w += dw;

    dh::ExecuteShards(&shards, [&](std::unique_ptr<DeviceShard> &shard) {
      shard->UpdateResidual(dw, group_idx, model->param.num_output_group, fidx);
    });
  }

  // training parameter
  GPUCoordinateTrainParam param;
  std::unique_ptr<FeatureSelector> selector;
  common::Monitor monitor;

  std::vector<std::unique_ptr<DeviceShard>> shards;
  std::vector<int> device_list;
};

DMLC_REGISTER_PARAMETER(GPUCoordinateTrainParam);
XGBOOST_REGISTER_LINEAR_UPDATER(GPUCoordinateUpdater, "gpu_coord_descent")
    .describe(
        "Update linear model according to coordinate descent algorithm. GPU "
        "accelerated.")
    .set_body([]() { return new GPUCoordinateUpdater(); });
}  // namespace linear
}  // namespace xgboost
