/*!
 * Copyright 2019 XGBoost contributors
 */

#include <xgboost/data.h>

#include "./ellpack_page.cuh"
#include "../common/hist_util.h"
#include "../common/random.h"

namespace xgboost {

EllpackPage::EllpackPage() = default;

EllpackPage::EllpackPage(DMatrix* dmat) : impl_{new EllpackPageImpl(dmat)} {}

EllpackPage::~EllpackPage() = default;

EllpackPageImpl::EllpackPageImpl(DMatrix* dmat) : dmat_{dmat} {}

// Bin each input data entry, store the bin indices in compressed form.
template<typename std::enable_if<true,  int>::type = 0>
__global__ void CompressBinEllpackKernel(
    common::CompressedBufferWriter wr,
    common::CompressedByteT* __restrict__ buffer,  // gidx_buffer
    const size_t* __restrict__ row_ptrs,           // row offset of input data
    const Entry* __restrict__ entries,      // One batch of input data
    const float* __restrict__ cuts,         // HistogramCuts::cut
    const uint32_t* __restrict__ cut_rows,  // HistogramCuts::row_ptrs
    size_t base_row,                        // batch_row_begin
    size_t n_rows,
    size_t row_stride,
    unsigned int null_gidx_value) {
  size_t irow = threadIdx.x + blockIdx.x * blockDim.x;
  int ifeature = threadIdx.y + blockIdx.y * blockDim.y;
  if (irow >= n_rows || ifeature >= row_stride) {
    return;
  }
  int row_length = static_cast<int>(row_ptrs[irow + 1] - row_ptrs[irow]);
  unsigned int bin = null_gidx_value;
  if (ifeature < row_length) {
    Entry entry = entries[row_ptrs[irow] - row_ptrs[0] + ifeature];
    int feature = entry.index;
    float fvalue = entry.fvalue;
    // {feature_cuts, ncuts} forms the array of cuts of `feature'.
    const float *feature_cuts = &cuts[cut_rows[feature]];
    int ncuts = cut_rows[feature + 1] - cut_rows[feature];
    // Assigning the bin in current entry.
    // S.t.: fvalue < feature_cuts[bin]
    bin = dh::UpperBound(feature_cuts, ncuts, fvalue);
    if (bin >= ncuts) {
      bin = ncuts - 1;
    }
    // Add the number of bins in previous features.
    bin += cut_rows[feature];
  }
  // Write to gidx buffer.
  wr.AtomicWriteSymbol(buffer, bin, (irow + base_row) * row_stride + ifeature);
}

void EllpackPageImpl::Init(int device, int max_bin, int gpu_batch_nrows) {
  if (initialised_) return;

  monitor_.Init("ellpack_page");
  dh::safe_cuda(cudaSetDevice(device));

  monitor_.StartCuda("Quantiles");
  // Create the quantile sketches for the dmatrix and initialize HistogramCuts.
  common::HistogramCuts hmat;
  size_t row_stride = common::DeviceSketch(device, max_bin, gpu_batch_nrows, dmat_, &hmat);
  monitor_.StopCuda("Quantiles");

  const auto& info = dmat_->Info();
  auto is_dense = info.num_nonzero_ == info.num_row_ * info.num_col_;

  // Init global data for each shard
  monitor_.StartCuda("InitCompressedData");
  InitCompressedData(device, hmat, row_stride, is_dense);
  monitor_.StopCuda("InitCompressedData");

  monitor_.StartCuda("BinningCompression");
  DeviceHistogramBuilderState hist_builder_row_state(info.num_row_);
  for (const auto& batch : dmat_->GetBatches<SparsePage>()) {
    hist_builder_row_state.BeginBatch(batch);
    CreateHistIndices(device, batch, hist_builder_row_state.GetRowStateOnDevice());
    hist_builder_row_state.EndBatch();
  }
  monitor_.StopCuda("BinningCompression");

  initialised_ = true;
}

void EllpackPageImpl::InitCompressedData(int device,
                                         const common::HistogramCuts& hmat,
                                         size_t row_stride,
                                         bool is_dense) {
  n_bins = hmat.Ptrs().back();
  int null_gidx_value = hmat.Ptrs().back();
  int num_symbols = n_bins + 1;

  // minimum value for each feature.
  common::Span<bst_float> min_fvalue;

  // Required buffer size for storing data matrix in ELLPack format.
  size_t compressed_size_bytes = common::CompressedBufferWriter::CalculateBufferSize(
      row_stride * dmat_->Info().num_row_, num_symbols);

  ba.Allocate(device,
              &feature_segments, hmat.Ptrs().size(),
              &gidx_fvalue_map, hmat.Values().size(),
              &min_fvalue, hmat.MinValues().size(),
              &gidx_buffer, compressed_size_bytes);

  dh::CopyVectorToDeviceSpan(gidx_fvalue_map, hmat.Values());
  dh::CopyVectorToDeviceSpan(min_fvalue, hmat.MinValues());
  dh::CopyVectorToDeviceSpan(feature_segments, hmat.Ptrs());
  thrust::fill(
      thrust::device_pointer_cast(gidx_buffer.data()),
      thrust::device_pointer_cast(gidx_buffer.data() + gidx_buffer.size()), 0);

  ellpack_matrix.Init(feature_segments,
                      min_fvalue,
                      gidx_fvalue_map,
                      row_stride,
                      common::CompressedIterator<uint32_t>(gidx_buffer.data(), num_symbols),
                      is_dense,
                      null_gidx_value);
}

void EllpackPageImpl::CreateHistIndices(int device,
                                        const SparsePage& row_batch,
                                        const RowStateOnDevice& device_row_state) {
  // Has any been allocated for me in this batch?
  if (!device_row_state.rows_to_process_from_batch) return;

  unsigned int null_gidx_value = n_bins;
  size_t row_stride = this->ellpack_matrix.row_stride;

  const auto &offset_vec = row_batch.offset.ConstHostVector();

  int num_symbols = n_bins + 1;
  // bin and compress entries in batches of rows
  size_t gpu_batch_nrows = std::min(
      dh::TotalMemory(device) / (16 * row_stride * sizeof(Entry)),
      static_cast<size_t>(device_row_state.rows_to_process_from_batch));
  const std::vector<Entry>& data_vec = row_batch.data.ConstHostVector();

  size_t gpu_nbatches = common::DivRoundUp(device_row_state.rows_to_process_from_batch,
                                           gpu_batch_nrows);

  for (size_t gpu_batch = 0; gpu_batch < gpu_nbatches; ++gpu_batch) {
    size_t batch_row_begin = gpu_batch * gpu_batch_nrows;
    size_t batch_row_end = (gpu_batch + 1) * gpu_batch_nrows;
    if (batch_row_end > device_row_state.rows_to_process_from_batch) {
      batch_row_end = device_row_state.rows_to_process_from_batch;
    }
    size_t batch_nrows = batch_row_end - batch_row_begin;

    const auto ent_cnt_begin =
        offset_vec[device_row_state.row_offset_in_current_batch + batch_row_begin];
    const auto ent_cnt_end =
        offset_vec[device_row_state.row_offset_in_current_batch + batch_row_end];

    /*! \brief row offset in SparsePage (the input data). */
    dh::device_vector<size_t> row_ptrs(batch_nrows+1);
    thrust::copy(
        offset_vec.data() + device_row_state.row_offset_in_current_batch + batch_row_begin,
        offset_vec.data() + device_row_state.row_offset_in_current_batch + batch_row_end + 1,
        row_ptrs.begin());

    // number of entries in this batch.
    size_t n_entries = ent_cnt_end - ent_cnt_begin;
    dh::device_vector<Entry> entries_d(n_entries);
    // copy data entries to device.
    dh::safe_cuda(cudaMemcpy(entries_d.data().get(),
                             data_vec.data() + ent_cnt_begin,
                             n_entries * sizeof(Entry),
                             cudaMemcpyDefault));
    const dim3 block3(32, 8, 1);  // 256 threads
    const dim3 grid3(common::DivRoundUp(batch_nrows, block3.x),
                     common::DivRoundUp(row_stride, block3.y),
                     1);
    CompressBinEllpackKernel<<<grid3, block3>>>(
        common::CompressedBufferWriter(num_symbols),
        gidx_buffer.data(),
        row_ptrs.data().get(),
        entries_d.data().get(),
        gidx_fvalue_map.data(),
        feature_segments.data(),
        device_row_state.total_rows_processed + batch_row_begin,
        batch_nrows,
        row_stride,
        null_gidx_value);
  }
}

}  // namespace xgboost
