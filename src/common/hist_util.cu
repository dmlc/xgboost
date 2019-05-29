/*!
 * Copyright 2018 XGBoost contributors
 */

#include "./hist_util.h"

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <utility>
#include <vector>
#include <memory>
#include <mutex>

#include "../tree/param.h"
#include "./host_device_vector.h"
#include "./device_helpers.cuh"
#include "./quantile.h"

namespace xgboost {
namespace common {

using WXQSketch = HistCutMatrix::WXQSketch;

__global__ void FindCutsK
(WXQSketch::Entry* __restrict__ cuts, const bst_float* __restrict__ data,
 const float* __restrict__ cum_weights, int nsamples, int ncuts) {
  // ncuts < nsamples
  int icut = threadIdx.x + blockIdx.x * blockDim.x;
  if (icut >= ncuts) {
    return;
  }
  WXQSketch::Entry v;
  int isample = 0;
  if (icut == 0) {
    isample = 0;
  } else if (icut == ncuts - 1) {
    isample = nsamples - 1;
  } else {
    bst_float rank = cum_weights[nsamples - 1] / static_cast<float>(ncuts - 1)
      * static_cast<float>(icut);
    // -1 is used because cum_weights is an inclusive sum
    isample = dh::UpperBound(cum_weights, nsamples, rank);
    isample = max(0, min(isample, nsamples - 1));
  }
  // repeated values will be filtered out on the CPU
  bst_float rmin = isample > 0 ? cum_weights[isample - 1] : 0;
  bst_float rmax = cum_weights[isample];
  cuts[icut] = WXQSketch::Entry(rmin, rmax, rmax - rmin, data[isample]);
}

// predictate for thrust filtering that returns true if the element is not a NaN
struct IsNotNaN {
  __device__ bool operator()(float a) const { return !isnan(a); }
};

__global__ void UnpackFeaturesK
(float* __restrict__ fvalues, float* __restrict__ feature_weights,
 const size_t* __restrict__ row_ptrs, const float* __restrict__ weights,
 Entry* entries, size_t nrows_array, int ncols, size_t row_begin_ptr,
 size_t nrows) {
  size_t irow = threadIdx.x + size_t(blockIdx.x) * blockDim.x;
  if (irow >= nrows) {
    return;
  }
  size_t row_length = row_ptrs[irow + 1] - row_ptrs[irow];
  int icol = threadIdx.y + blockIdx.y * blockDim.y;
  if (icol >= row_length) {
    return;
  }
  Entry entry = entries[row_ptrs[irow] - row_begin_ptr + icol];
  size_t ind = entry.index * nrows_array + irow;
  // if weights are present, ensure that a non-NaN value is written to weights
  // if and only if it is also written to features
  if (!isnan(entry.fvalue) && (weights == nullptr || !isnan(weights[irow]))) {
    fvalues[ind] = entry.fvalue;
    if (feature_weights != nullptr && weights != nullptr) {
      feature_weights[ind] = weights[irow];
    }
  }
}

/*!
 * \brief A container that holds the device sketches across all
 *  sparse page batches which are distributed to different devices.
 *  As sketches are aggregated by column, the mutex guards
 *  multiple devices pushing sketch summary for the same column
 *  across distinct rows.
 */
struct SketchContainer {
  std::vector<HistCutMatrix::WXQSketch> sketches_;  // NOLINT
  std::vector<std::mutex> col_locks_; // NOLINT
  static constexpr int kOmpNumColsParallelizeLimit = 1000;

  SketchContainer(const tree::TrainParam &param, DMatrix *dmat) :
    col_locks_(dmat->Info().num_col_) {
    const MetaInfo &info = dmat->Info();
    // Initialize Sketches for this dmatrix
    sketches_.resize(info.num_col_);
#pragma omp parallel for schedule(static) if (info.num_col_ > kOmpNumColsParallelizeLimit)
    for (int icol = 0; icol < info.num_col_; ++icol) {
      sketches_[icol].Init(info.num_row_, 1.0 / (8 * param.max_bin));
    }
  }

  // Prevent copying/assigning/moving this as its internals can't be assigned/copied/moved
  SketchContainer(const SketchContainer &) = delete;
  SketchContainer(const SketchContainer &&) = delete;
  SketchContainer &operator=(const SketchContainer &) = delete;
  SketchContainer &operator=(const SketchContainer &&) = delete;
};

// finds quantiles on the GPU
struct GPUSketcher {
  // manage memory for a single GPU
  class DeviceShard {
    int device_;
    bst_uint row_begin_;  // The row offset for this shard
    bst_uint row_end_;
    bst_uint n_rows_;
    int num_cols_{0};
    size_t n_cuts_{0};
    size_t gpu_batch_nrows_{0};
    bool has_weights_{false};
    size_t row_stride_{0};

    tree::TrainParam param_;
    SketchContainer *sketch_container_;
    thrust::device_vector<size_t> row_ptrs_;
    thrust::device_vector<Entry> entries_;
    thrust::device_vector<bst_float> fvalues_;
    thrust::device_vector<bst_float> feature_weights_;
    thrust::device_vector<bst_float> fvalues_cur_;
    thrust::device_vector<WXQSketch::Entry> cuts_d_;
    thrust::host_vector<WXQSketch::Entry> cuts_h_;
    thrust::device_vector<bst_float> weights_;
    thrust::device_vector<bst_float> weights2_;
    std::vector<size_t> n_cuts_cur_;
    thrust::device_vector<size_t> num_elements_;
    thrust::device_vector<char> tmp_storage_;

   public:
    DeviceShard(int device, bst_uint row_begin, bst_uint row_end,
                tree::TrainParam param, SketchContainer *sketch_container) :
      device_(device), row_begin_(row_begin), row_end_(row_end),
      n_rows_(row_end - row_begin), param_(std::move(param)), sketch_container_(sketch_container) {
    }

    inline size_t GetRowStride() const {
       return row_stride_;
    }

    void Init(const SparsePage& row_batch, const MetaInfo& info, int gpu_batch_nrows) {
      num_cols_ = info.num_col_;
      has_weights_ = info.weights_.Size() > 0;

      // find the batch size
      if (gpu_batch_nrows == 0) {
        // By default, use no more than 1/16th of GPU memory
        gpu_batch_nrows_ = dh::TotalMemory(device_) /
          (16 * num_cols_ * sizeof(Entry));
      } else if (gpu_batch_nrows == -1) {
        gpu_batch_nrows_ = n_rows_;
      } else {
        gpu_batch_nrows_ = gpu_batch_nrows;
      }
      if (gpu_batch_nrows_ > n_rows_) {
        gpu_batch_nrows_ = n_rows_;
      }

      constexpr int kFactor = 8;
      double eps = 1.0 / (kFactor * param_.max_bin);
      size_t dummy_nlevel;
      WXQSketch::LimitSizeLevel(gpu_batch_nrows_, eps, &dummy_nlevel, &n_cuts_);

      // allocate necessary GPU buffers
      dh::safe_cuda(cudaSetDevice(device_));

      entries_.resize(gpu_batch_nrows_ * num_cols_);
      fvalues_.resize(gpu_batch_nrows_ * num_cols_);
      fvalues_cur_.resize(gpu_batch_nrows_);
      cuts_d_.resize(n_cuts_ * num_cols_);
      cuts_h_.resize(n_cuts_ * num_cols_);
      weights_.resize(gpu_batch_nrows_);
      weights2_.resize(gpu_batch_nrows_);
      num_elements_.resize(1);

      if (has_weights_) {
        feature_weights_.resize(gpu_batch_nrows_ * num_cols_);
      }
      n_cuts_cur_.resize(num_cols_);

      // allocate storage for CUB algorithms; the size is the maximum of the sizes
      // required for various algorithm
      size_t tmp_size = 0, cur_tmp_size = 0;
      // size for sorting
      if (has_weights_) {
        cub::DeviceRadixSort::SortPairs
          (nullptr, cur_tmp_size, fvalues_cur_.data().get(),
           fvalues_.data().get(), weights_.data().get(), weights2_.data().get(),
           gpu_batch_nrows_);
      } else {
        cub::DeviceRadixSort::SortKeys
          (nullptr, cur_tmp_size, fvalues_cur_.data().get(), fvalues_.data().get(),
           gpu_batch_nrows_);
      }
      tmp_size = std::max(tmp_size, cur_tmp_size);
      // size for inclusive scan
      if (has_weights_) {
        cub::DeviceScan::InclusiveSum
          (nullptr, cur_tmp_size, weights2_.begin(), weights_.begin(), gpu_batch_nrows_);
        tmp_size = std::max(tmp_size, cur_tmp_size);
      }
      // size for reduction by key
      cub::DeviceReduce::ReduceByKey
        (nullptr, cur_tmp_size, fvalues_.begin(),
         fvalues_cur_.begin(), weights_.begin(), weights2_.begin(),
         num_elements_.begin(), thrust::maximum<bst_float>(), gpu_batch_nrows_);
      tmp_size = std::max(tmp_size, cur_tmp_size);
      // size for filtering
      cub::DeviceSelect::If
        (nullptr, cur_tmp_size, fvalues_.begin(), fvalues_cur_.begin(),
         num_elements_.begin(), gpu_batch_nrows_, IsNotNaN());
      tmp_size = std::max(tmp_size, cur_tmp_size);

      tmp_storage_.resize(tmp_size);
    }

    void FindColumnCuts(size_t batch_nrows, size_t icol) {
      size_t tmp_size = tmp_storage_.size();
      // filter out NaNs in feature values
      auto fvalues_begin = fvalues_.data() + icol * gpu_batch_nrows_;
      cub::DeviceSelect::If
        (tmp_storage_.data().get(), tmp_size, fvalues_begin,
         fvalues_cur_.data(), num_elements_.begin(), batch_nrows, IsNotNaN());
      size_t nfvalues_cur = 0;
      thrust::copy_n(num_elements_.begin(), 1, &nfvalues_cur);

      // compute cumulative weights using a prefix scan
      if (has_weights_) {
        // filter out NaNs in weights;
        // since cub::DeviceSelect::If performs stable filtering,
        // the weights are stored in the correct positions
        auto feature_weights_begin = feature_weights_.data() +
          icol * gpu_batch_nrows_;
        cub::DeviceSelect::If
          (tmp_storage_.data().get(), tmp_size, feature_weights_begin,
           weights_.data().get(), num_elements_.begin(), batch_nrows, IsNotNaN());

        // sort the values and weights
        cub::DeviceRadixSort::SortPairs
          (tmp_storage_.data().get(), tmp_size, fvalues_cur_.data().get(),
           fvalues_begin.get(), weights_.data().get(), weights2_.data().get(),
           nfvalues_cur);

        // sum the weights to get cumulative weight values
        cub::DeviceScan::InclusiveSum
          (tmp_storage_.data().get(), tmp_size, weights2_.begin(),
           weights_.begin(), nfvalues_cur);
      } else {
        // sort the batch values
        cub::DeviceRadixSort::SortKeys
          (tmp_storage_.data().get(), tmp_size,
           fvalues_cur_.data().get(), fvalues_begin.get(), nfvalues_cur);

        // fill in cumulative weights with counting iterator
        thrust::copy_n(thrust::make_counting_iterator(1), nfvalues_cur,
                       weights_.begin());
      }

      // remove repeated items and sum the weights across them;
      // non-negative weights are assumed
      cub::DeviceReduce::ReduceByKey
        (tmp_storage_.data().get(), tmp_size, fvalues_begin,
         fvalues_cur_.begin(), weights_.begin(), weights2_.begin(),
         num_elements_.begin(), thrust::maximum<bst_float>(), nfvalues_cur);
      size_t n_unique = 0;
      thrust::copy_n(num_elements_.begin(), 1, &n_unique);

      // extract cuts
      n_cuts_cur_[icol] = std::min(n_cuts_, n_unique);
      // if less elements than cuts: copy all elements with their weights
      if (n_cuts_ > n_unique) {
        float* weights2_ptr = weights2_.data().get();
        float* fvalues_ptr = fvalues_cur_.data().get();
        WXQSketch::Entry* cuts_ptr = cuts_d_.data().get() + icol * n_cuts_;
        dh::LaunchN(device_, n_unique, [=]__device__(size_t i) {
            bst_float rmax = weights2_ptr[i];
            bst_float rmin = i > 0 ? weights2_ptr[i - 1] : 0;
            cuts_ptr[i] = WXQSketch::Entry(rmin, rmax, rmax - rmin, fvalues_ptr[i]);
          });
      } else if (n_cuts_cur_[icol] > 0) {
        // if more elements than cuts: use binary search on cumulative weights
        int block = 256;
        FindCutsK<<<dh::DivRoundUp(n_cuts_cur_[icol], block), block>>>
          (cuts_d_.data().get() + icol * n_cuts_, fvalues_cur_.data().get(),
           weights2_.data().get(), n_unique, n_cuts_cur_[icol]);
        dh::safe_cuda(cudaGetLastError());  // NOLINT
      }
    }

    void SketchBatch(const SparsePage& row_batch, const MetaInfo& info,
                     size_t gpu_batch) {
      // compute start and end indices
      size_t batch_row_begin = gpu_batch * gpu_batch_nrows_;
      size_t batch_row_end = std::min((gpu_batch + 1) * gpu_batch_nrows_,
                                      static_cast<size_t>(n_rows_));
      size_t batch_nrows = batch_row_end - batch_row_begin;

      const auto& offset_vec = row_batch.offset.HostVector();
      const auto& data_vec = row_batch.data.HostVector();

      size_t n_entries = offset_vec[row_begin_ + batch_row_end] -
        offset_vec[row_begin_ + batch_row_begin];
      // copy the batch to the GPU
      dh::safe_cuda
        (cudaMemcpyAsync(entries_.data().get(),
                    data_vec.data() + offset_vec[row_begin_ + batch_row_begin],
                    n_entries * sizeof(Entry), cudaMemcpyDefault));
      // copy the weights if necessary
      if (has_weights_) {
        const auto& weights_vec = info.weights_.HostVector();
        dh::safe_cuda
          (cudaMemcpyAsync(weights_.data().get(),
                      weights_vec.data() + row_begin_ + batch_row_begin,
                      batch_nrows * sizeof(bst_float), cudaMemcpyDefault));
      }

      // unpack the features; also unpack weights if present
      thrust::fill(fvalues_.begin(), fvalues_.end(), NAN);
      if (has_weights_) {
        thrust::fill(feature_weights_.begin(), feature_weights_.end(), NAN);
      }

      dim3 block3(16, 64, 1);
      // NOTE: This will typically support ~ 4M features - 64K*64
      dim3 grid3(dh::DivRoundUp(batch_nrows, block3.x),
                 dh::DivRoundUp(num_cols_, block3.y), 1);
      UnpackFeaturesK<<<grid3, block3>>>
        (fvalues_.data().get(), has_weights_ ? feature_weights_.data().get() : nullptr,
         row_ptrs_.data().get() + batch_row_begin,
         has_weights_ ? weights_.data().get() : nullptr, entries_.data().get(),
         gpu_batch_nrows_, num_cols_,
         offset_vec[row_begin_ + batch_row_begin], batch_nrows);

      for (int icol = 0; icol < num_cols_; ++icol) {
        FindColumnCuts(batch_nrows, icol);
      }

      // add cuts into sketches
      thrust::copy(cuts_d_.begin(), cuts_d_.end(), cuts_h_.begin());
#pragma omp parallel for schedule(static) \
      if (num_cols_ > SketchContainer::kOmpNumColsParallelizeLimit) // NOLINT
      for (int icol = 0; icol < num_cols_; ++icol) {
        WXQSketch::SummaryContainer summary;
        summary.Reserve(n_cuts_);
        summary.MakeFromSorted(&cuts_h_[n_cuts_ * icol], n_cuts_cur_[icol]);

        std::lock_guard<std::mutex> lock(sketch_container_->col_locks_[icol]);
        sketch_container_->sketches_[icol].PushSummary(summary);
      }
    }

    void ComputeRowStride() {
      // Find the row stride for this batch
      auto row_iter = row_ptrs_.begin();
      // Functor for finding the maximum row size for this batch
      auto get_size = [=] __device__(size_t row) {
        return row_iter[row + 1] - row_iter[row];
      }; // NOLINT

      auto counting = thrust::make_counting_iterator(size_t(0));
      using TransformT = thrust::transform_iterator<decltype(get_size),
                                                    decltype(counting), size_t>;
      TransformT row_size_iter = TransformT(counting, get_size);
      row_stride_ = thrust::reduce(row_size_iter, row_size_iter + n_rows_, 0,
                                   thrust::maximum<size_t>());
    }

    void Sketch(const SparsePage& row_batch, const MetaInfo& info) {
      // copy rows to the device
      dh::safe_cuda(cudaSetDevice(device_));
      const auto& offset_vec = row_batch.offset.HostVector();
      row_ptrs_.resize(n_rows_ + 1);
      thrust::copy(offset_vec.data() + row_begin_,
                   offset_vec.data() + row_end_ + 1, row_ptrs_.begin());
      size_t gpu_nbatches = dh::DivRoundUp(n_rows_, gpu_batch_nrows_);
      for (size_t gpu_batch = 0; gpu_batch < gpu_nbatches; ++gpu_batch) {
        SketchBatch(row_batch, info, gpu_batch);
      }
    }
  };

  void SketchBatch(const SparsePage &batch, const MetaInfo &info) {
    GPUDistribution dist =
      GPUDistribution::Block(GPUSet::All(learner_param_.gpu_id, learner_param_.n_gpus,
                                         batch.Size()));

    // create device shards
    shards_.resize(dist.Devices().Size());
    dh::ExecuteIndexShards(&shards_, [&](int i, std::unique_ptr<DeviceShard>& shard) {
        size_t start = dist.ShardStart(batch.Size(), i);
        size_t size = dist.ShardSize(batch.Size(), i);
        shard = std::unique_ptr<DeviceShard>(
            new DeviceShard(dist.Devices().DeviceId(i), start,
                            start + size, param_, sketch_container_.get()));
      });

    // compute sketches for each shard
    dh::ExecuteIndexShards(&shards_,
                           [&](int idx, std::unique_ptr<DeviceShard>& shard) {
                             shard->Init(batch, info, gpu_batch_nrows_);
                             shard->Sketch(batch, info);
                             shard->ComputeRowStride();
                           });

    // compute row stride across all shards
    for (const auto &shard : shards_) {
      row_stride_ = std::max(row_stride_, shard->GetRowStride());
    }
  }

  GPUSketcher(const tree::TrainParam &param, const LearnerTrainParam &learner_param, int gpu_nrows)
    : param_(param), learner_param_(learner_param), gpu_batch_nrows_(gpu_nrows), row_stride_(0) {
  }

  /* Builds the sketches on the GPU for the dmatrix and returns the row stride
   * for the entire dataset */
  size_t Sketch(DMatrix *dmat, HistCutMatrix *hmat) {
    const MetaInfo &info = dmat->Info();

    row_stride_ = 0;
    sketch_container_.reset(new SketchContainer(param_, dmat));
    for (const auto &batch : dmat->GetRowBatches()) {
      this->SketchBatch(batch, info);
    }

    hmat->Init(&sketch_container_.get()->sketches_, param_.max_bin);

    return row_stride_;
  }

 private:
  std::vector<std::unique_ptr<DeviceShard>> shards_;
  const tree::TrainParam &param_;
  const LearnerTrainParam &learner_param_;
  int gpu_batch_nrows_;
  size_t row_stride_;
  std::unique_ptr<SketchContainer> sketch_container_;
};

size_t DeviceSketch
  (const tree::TrainParam &param, const LearnerTrainParam &learner_param, int gpu_batch_nrows,
   DMatrix *dmat, HistCutMatrix *hmat) {
  GPUSketcher sketcher(param, learner_param, gpu_batch_nrows);
  return sketcher.Sketch(dmat, hmat);
}

}  // namespace common
}  // namespace xgboost
