/*!
 * Copyright 2017 by Contributors
 * \file hist_util.cu
 * \brief Finding quantiles ("sketching") on the GPU
 * \author Andy Adinets
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

#include "../tree/param.h"
#include "./host_device_vector.h"
#include "./device_helpers.cuh"
#include "./quantile.h"

namespace xgboost {
namespace common {

using WXQSketch = HistCutMatrix::WXQSketch;

__global__ void find_cuts_k
(WXQSketch::Entry* __restrict__ cuts, const bst_float* __restrict__ data,
 const float* __restrict__ cum_weights, int nsamples, int ncuts) {
  // ncuts < nsamples
  int icut = threadIdx.x + blockIdx.x * blockDim.x;
  if (icut >= ncuts)
    return;
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
    isample = dh::UpperBound(cum_weights, nsamples, rank) - 1;
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

__global__ void unpack_features_k
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
    if (feature_weights != nullptr) {
      feature_weights[ind] = weights[irow];
    }
  }
}

// finds quantiles on the GPU
struct GPUSketcher {
  // manage memory for a single GPU
  struct DeviceShard {
    int device_;
    std::vector<WXQSketch> sketches_;
    thrust::device_vector<size_t> row_ptrs_;
    bst_uint row_begin_;  // The row offset for this shard
    bst_uint row_end_;
    bst_uint n_rows_;
    int gpu_batch_nrows_;

    DeviceShard(int device, bst_uint row_begin, bst_uint row_end, int gpu_batch_nrows) :
      device_(device), row_begin_(row_begin), row_end_(row_end),
      n_rows_(row_end - row_begin), gpu_batch_nrows_(gpu_batch_nrows) {}

    size_t BatchSize(size_t max_num_cols) {
      size_t gpu_batch_nrows = gpu_batch_nrows_;
      if (gpu_batch_nrows_ == 0) {
        // By default, use no more than 1/16th of GPU memory
        gpu_batch_nrows = dh::TotalMemory(device_) /
          (16 * max_num_cols * sizeof(Entry));
      } else if (gpu_batch_nrows_ == -1) {
        gpu_batch_nrows = n_rows_;
      }
      if (gpu_batch_nrows > n_rows_) {
        gpu_batch_nrows = n_rows_;
      }
      return gpu_batch_nrows;
    }

    void Sketch(const SparsePage& row_batch, const MetaInfo& info, int max_num_bins) {
      // copy rows to the device
      dh::safe_cuda(cudaSetDevice(device_));
      row_ptrs_.resize(n_rows_ + 1);
      thrust::copy(&row_batch.offset[row_begin_], &row_batch.offset[row_end_ + 1],
                   row_ptrs_.begin());

      std::vector<WXQSketch::SummaryContainer> summaries;
      // initialize sketches
      size_t num_cols = info.num_col_;
      sketches_.resize(num_cols);
      summaries.resize(num_cols);
      constexpr int kFactor = 8;
      double eps = 1.0 / (kFactor * max_num_bins);
      size_t dummy_nlevel;
      size_t ncuts = 0;
      WXQSketch::LimitSizeLevel(row_batch.Size(), eps, &dummy_nlevel, &ncuts);
      // double ncuts to be the same as the number of values
      // in the temporary buffers of the sketches
      ncuts *= 2;
      for (int icol = 0; icol < num_cols; ++icol) {
        sketches_[icol].Init(row_batch.Size(), eps);
        summaries[icol].Reserve(ncuts);
      }

      // allocate necessary GPU buffers
      dh::safe_cuda(cudaSetDevice(device_));

      size_t gpu_batch_nrows = BatchSize(num_cols);
      thrust::device_vector<Entry> entries(gpu_batch_nrows * num_cols);
      thrust::device_vector<bst_float> fvalues(gpu_batch_nrows * num_cols);
      thrust::device_vector<bst_float> feature_weights;
      thrust::device_vector<bst_float> fvalues_cur(gpu_batch_nrows);
      thrust::device_vector<WXQSketch::Entry> cuts_d(ncuts * num_cols);
      thrust::host_vector<WXQSketch::Entry> cuts_h(ncuts * num_cols);
      thrust::device_vector<bst_float> weights(gpu_batch_nrows);
      thrust::device_vector<bst_float> weights2(gpu_batch_nrows);
      bool has_weights = info.weights_.size() > 0;
      if (has_weights) {
        feature_weights.resize(gpu_batch_nrows * num_cols);
      }
      std::vector<size_t> ncuts_cur(num_cols);

      // temporary storage for number of elements for filtering and reduction using CUB
      thrust::device_vector<size_t> num_elements(1);

      // temporary storage for sorting using CUB
      size_t sort_tmp_size = 0;
      if (has_weights) {
        cub::DeviceRadixSort::SortPairs
          (nullptr, sort_tmp_size, fvalues_cur.data().get(),
           fvalues.data().get(), weights.data().get(), weights2.data().get(),
           gpu_batch_nrows);
      } else {
        cub::DeviceRadixSort::SortKeys
          (nullptr, sort_tmp_size, fvalues_cur.data().get(), fvalues.data().get(),
           gpu_batch_nrows);
      }
      thrust::device_vector<char> sort_tmp_storage(sort_tmp_size);

      // temporary storage for inclusive prefix sum using CUB
      size_t scan_tmp_size = 0;
      if (has_weights) {
        cub::DeviceScan::InclusiveSum
          (nullptr, scan_tmp_size, weights2.begin(), weights.begin(), gpu_batch_nrows);
      }
      thrust::device_vector<char> scan_tmp_storage(scan_tmp_size);

      // temporary storage for reduction by key using CUB
      size_t reduce_tmp_size = 0;
      cub::DeviceReduce::ReduceByKey
        (nullptr, reduce_tmp_size, fvalues.begin(),
         fvalues_cur.begin(), weights.begin(), weights2.begin(), num_elements.begin(),
         thrust::maximum<bst_float>(), gpu_batch_nrows);
      thrust::device_vector<char> reduce_tmp_storage(reduce_tmp_size);

      // temporary storage for filtering using CUB
      size_t if_tmp_size = 0;
      cub::DeviceSelect::If(nullptr, if_tmp_size, fvalues.begin(), fvalues_cur.begin(),
                            num_elements.begin(), gpu_batch_nrows, IsNotNaN());
      thrust::device_vector<char> if_tmp_storage(if_tmp_size);

      size_t gpu_nbatches = dh::DivRoundUp(n_rows_, gpu_batch_nrows);

      for (size_t gpu_batch = 0; gpu_batch < gpu_nbatches; ++gpu_batch) {
        // compute start and end indices
        size_t batch_row_begin = gpu_batch * gpu_batch_nrows;
        size_t batch_row_end = std::min((gpu_batch + 1) * gpu_batch_nrows,
                                        static_cast<size_t>(n_rows_));
        size_t batch_nrows = batch_row_end - batch_row_begin;
        size_t n_entries =
          row_batch.offset[row_begin_ + batch_row_end] -
          row_batch.offset[row_begin_ + batch_row_begin];
        // copy the batch to the GPU
        dh::safe_cuda
          (cudaMemcpy(entries.data().get(),
                      &row_batch.data[row_batch.offset[row_begin_ + batch_row_begin]],
                      n_entries * sizeof(Entry), cudaMemcpyDefault));
        // copy the weights if necessary
        if (has_weights) {
          dh::safe_cuda
            (cudaMemcpy(weights.data().get(),
                        info.weights_.data() + row_begin_ + batch_row_begin,
                        batch_nrows * sizeof(bst_float), cudaMemcpyDefault));
        }

        // unpack the features; also unpack weights if present
        thrust::fill(fvalues.begin(), fvalues.end(), NAN);
        thrust::fill(feature_weights.begin(), feature_weights.end(), NAN);

        dim3 block3(64, 4, 1);
        dim3 grid3(dh::DivRoundUp(batch_nrows, block3.x),
                   dh::DivRoundUp(num_cols, block3.y), 1);
        unpack_features_k<<<grid3, block3>>>
          (fvalues.data().get(), has_weights ? feature_weights.data().get() : nullptr,
           row_ptrs_.data().get() + batch_row_begin,
           has_weights ? weights.data().get() : nullptr, entries.data().get(),
           gpu_batch_nrows, num_cols,
           row_batch.offset[row_begin_ + batch_row_begin], batch_nrows);
        dh::safe_cuda(cudaGetLastError());
        dh::safe_cuda(cudaDeviceSynchronize());

        for (int icol = 0; icol < num_cols; ++icol) {
          // filter out NaNs in feature values
          auto fvalues_begin = fvalues.data() + icol * gpu_batch_nrows;
          cub::DeviceSelect::If
            (if_tmp_storage.data().get(), if_tmp_size, fvalues_begin, fvalues_cur.data(),
             num_elements.begin(), batch_nrows, IsNotNaN());
          size_t nfvalues_cur = 0;
          thrust::copy_n(num_elements.begin(), 1, &nfvalues_cur);

          // compute cumulative weights using a prefix scan
          if (has_weights) {
            // filter out NaNs in weights;
            // since cub::DeviceSelect::If performs stable filtering,
            // the weights are stored in the correct positions
            auto feature_weights_begin = feature_weights.data() + icol * gpu_batch_nrows;
            cub::DeviceSelect::If
              (if_tmp_storage.data().get(), if_tmp_size, feature_weights_begin,
               weights.data().get(), num_elements.begin(), batch_nrows, IsNotNaN());

            // sort the values and weights
            cub::DeviceRadixSort::SortPairs
              (sort_tmp_storage.data().get(), sort_tmp_size, fvalues_cur.data().get(),
               fvalues_begin.get(), weights.data().get(), weights2.data().get(), nfvalues_cur);

            // sum the weights to get cumulative weight values
            cub::DeviceScan::InclusiveSum
              (scan_tmp_storage.data().get(), scan_tmp_size, weights2.begin(),
               weights.begin(), nfvalues_cur);
          } else {
            // sort the batch values
            cub::DeviceRadixSort::SortKeys
              (sort_tmp_storage.data().get(), sort_tmp_size,
               fvalues_cur.data().get(), fvalues_begin.get(), nfvalues_cur);

            // fill in cumulative weights with counting iterator
            thrust::copy_n(thrust::make_counting_iterator(1), nfvalues_cur,
                           weights.begin());
          }

          // remove repeated items and sum the weights across them;
          // non-negative weights are assumed
          cub::DeviceReduce::ReduceByKey
            (reduce_tmp_storage.data().get(), reduce_tmp_size, fvalues_begin,
             fvalues_cur.begin(), weights.begin(), weights2.begin(), num_elements.begin(),
             thrust::maximum<bst_float>(), nfvalues_cur);
          size_t n_unique = 0;
          thrust::copy_n(num_elements.begin(), 1, &n_unique);

          // extract cuts
          ncuts_cur[icol] = ncuts < n_unique ? ncuts : n_unique;
          // if less elements than cuts: copy all elements with their weights
          if (ncuts > n_unique) {
            auto weights2_iter = weights2.begin();
            auto fvalues_iter = fvalues_cur.begin();
            auto cuts_iter = cuts_d.begin() + icol * ncuts;
            dh::LaunchN(device_, n_unique, [=]__device__(size_t i) {
                bst_float rmax = weights2_iter[i];
                bst_float rmin = i > 0 ? weights2_iter[i - 1] : 0;
                cuts_iter[i] = WXQSketch::Entry(rmin, rmax, rmax - rmin, fvalues_iter[i]);
              });
          } else if (ncuts_cur[icol] > 0) {
            // if more elements than cuts: use binary search on cumulative weights
            int block = 256;
            find_cuts_k<<<dh::DivRoundUp(ncuts_cur[icol], block), block>>>
              (cuts_d.data().get() + icol * ncuts, fvalues_cur.data().get(),
               weights2.data().get(), n_unique, ncuts_cur[icol]);
            dh::safe_cuda(cudaGetLastError());
          }
        }

        dh::safe_cuda(cudaDeviceSynchronize());

        // add cuts into sketches
        thrust::copy(cuts_d.begin(), cuts_d.end(), cuts_h.begin());
        for (int icol = 0; icol < num_cols; ++icol) {
          summaries[icol].MakeFromSorted(&cuts_h[ncuts * icol], ncuts_cur[icol]);
          sketches_[icol].PushSummary(summaries[icol]);
        }
      }
    }
  };

  void Sketch(const SparsePage& batch, const MetaInfo& info, HistCutMatrix* hmat) {
    // partition input matrix into row segments
    std::vector<size_t> row_segments;
    row_segments.push_back(0);
    bst_uint row_begin = 0;
    bst_uint shard_size = dh::DivRoundUp(info.num_row_, devices_.Size());
    for (int d_idx = 0; d_idx < devices_.Size(); ++d_idx) {
      bst_uint row_end =
          std::min(static_cast<size_t>(row_begin + shard_size), info.num_row_);
      row_segments.push_back(row_end);
      row_begin = row_end;
    }

    // create device shards
    shards_.resize(devices_.Size());
    dh::ExecuteIndexShards(&shards_, [&](int i, std::unique_ptr<DeviceShard>& shard) {
        shard = std::unique_ptr<DeviceShard>
          (new DeviceShard(devices_[i], row_segments[i],
                           row_segments[i + 1], param_.gpu_batch_nrows));
      });

    // compute sketches for each shard
    dh::ExecuteShards(&shards_, [&](std::unique_ptr<DeviceShard>& shard) {
        shard->Sketch(batch, info, param_.max_bin);
      });

    // merge the sketches from all shards
    // TODO(canonizer): do it in a tree-like reduction
    int num_cols = info.num_col_;
    std::vector<WXQSketch> sketches(num_cols);
    WXQSketch::SummaryContainer summary;
    for (int icol = 0; icol < num_cols; ++icol) {
      sketches[icol].Init(batch.Size(), 1.0 / (8 * param_.max_bin));
      for (int shard = 0; shard < shards_.size(); ++shard) {
        shards_[shard]->sketches_[icol].GetSummary(&summary);
        sketches[icol].PushSummary(summary);
      }
    }

    hmat->Init(&sketches, param_.max_bin);
  }

  explicit GPUSketcher(const tree::TrainParam& param) : param_(param) {
    devices_ = GPUSet::Range(param_.gpu_id, dh::NDevicesAll(param_.n_gpus));
  }

  std::vector<std::unique_ptr<DeviceShard>> shards_;
  tree::TrainParam param_;
  GPUSet devices_;
};

void DeviceSketch
  (const SparsePage& batch, const MetaInfo& info,
   const tree::TrainParam& param, HistCutMatrix* hmat) {
  GPUSketcher sketcher(param);
  sketcher.Sketch(batch, info, hmat);
}

}  // namespace common
}  // namespace xgboost
