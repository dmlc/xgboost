/*!
 * Copyright 2019 XGBoost contributors
 */

#include <xgboost/data.h>

#include "./ellpack_page.cuh"
#include "../common/hist_util.h"
#include "../common/random.h"

namespace xgboost {

EllpackPage::EllpackPage() : impl_{new EllpackPageImpl()} {}

EllpackPage::EllpackPage(DMatrix* dmat, const BatchParam& param)
    : impl_{new EllpackPageImpl(dmat, param)} {}

EllpackPage::~EllpackPage() = default;

size_t EllpackPage::Size() const {
  return impl_->Size();
}

void EllpackPage::SetBaseRowId(size_t row_id) {
  impl_->SetBaseRowId(row_id);
}

// Bin each input data entry, store the bin indices in compressed form.
__global__ void CompressBinEllpackKernel(
    common::CompressedBufferWriter wr,
    common::CompressedByteT* __restrict__ buffer,  // gidx_buffer
    const size_t* __restrict__ row_ptrs,           // row offset of input data
    const Entry* __restrict__ entries,      // One batch of input data
    const float* __restrict__ cuts,         // HistogramCuts::cut_values_
    const uint32_t* __restrict__ cut_rows,  // HistogramCuts::cut_ptrs_
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
    const float* feature_cuts = &cuts[cut_rows[feature]];
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

// Construct an ELLPACK matrix with the given number of empty rows.
EllpackPageImpl::EllpackPageImpl(int device, EllpackInfo info, size_t n_rows) {
  monitor_.Init("ellpack_page");
  dh::safe_cuda(cudaSetDevice(device));

  matrix.info = info;
  matrix.base_rowid = 0;
  matrix.n_rows = n_rows;

  monitor_.StartCuda("InitCompressedData");
  InitCompressedData(device, n_rows);
  monitor_.StopCuda("InitCompressedData");
}

// Construct an ELLPACK matrix in memory.
EllpackPageImpl::EllpackPageImpl(DMatrix* dmat, const BatchParam& param) {
  monitor_.Init("ellpack_page");
  dh::safe_cuda(cudaSetDevice(param.gpu_id));

  matrix.n_rows = dmat->Info().num_row_;

  monitor_.StartCuda("Quantiles");
  // Create the quantile sketches for the dmatrix and initialize HistogramCuts.
  common::HistogramCuts hmat;
  size_t row_stride =
      common::DeviceSketch(param.gpu_id, param.max_bin, param.gpu_batch_nrows, dmat, &hmat);
  monitor_.StopCuda("Quantiles");

  monitor_.StartCuda("InitEllpackInfo");
  InitInfo(param.gpu_id, dmat->IsDense(), row_stride, hmat);
  monitor_.StopCuda("InitEllpackInfo");

  monitor_.StartCuda("InitCompressedData");
  InitCompressedData(param.gpu_id, dmat->Info().num_row_);
  monitor_.StopCuda("InitCompressedData");

  monitor_.StartCuda("BinningCompression");
  DeviceHistogramBuilderState hist_builder_row_state(dmat->Info().num_row_);
  for (const auto& batch : dmat->GetBatches<SparsePage>()) {
    hist_builder_row_state.BeginBatch(batch);
    CreateHistIndices(param.gpu_id, batch, hist_builder_row_state.GetRowStateOnDevice());
    hist_builder_row_state.EndBatch();
  }
  monitor_.StopCuda("BinningCompression");
}

// A functor that copies the data from one EllpackPage to another.
struct CopyPage {
  common::CompressedBufferWriter cbw;
  common::CompressedByteT* dst_data_d;
  common::CompressedIterator<uint32_t> src_iterator_d;
  // The number of elements to skip.
  size_t offset;

  CopyPage(EllpackPageImpl* dst, EllpackPageImpl* src, size_t offset)
      : cbw{dst->matrix.info.NumSymbols()},
        dst_data_d{dst->gidx_buffer.data()},
        src_iterator_d{src->gidx_buffer.data(), src->matrix.info.NumSymbols()},
        offset(offset) {}

  __device__ void operator()(size_t element_id) {
    cbw.AtomicWriteSymbol(dst_data_d, src_iterator_d[element_id], element_id + offset);
  }
};

// Copy the data from the given EllpackPage to the current page.
size_t EllpackPageImpl::Copy(int device, EllpackPageImpl* page, size_t offset) {
  monitor_.StartCuda("Copy");
  size_t num_elements = page->matrix.n_rows * page->matrix.info.row_stride;
  CHECK_EQ(matrix.info.row_stride, page->matrix.info.row_stride);
  CHECK_EQ(matrix.info.NumSymbols(), page->matrix.info.NumSymbols());
  CHECK_GE(matrix.n_rows * matrix.info.row_stride, offset + num_elements);
  dh::LaunchN(device, num_elements, CopyPage(this, page, offset));
  monitor_.StopCuda("Copy");
  return num_elements;
}

// A functor that compacts the rows from one EllpackPage into another.
struct CompactPage {
  common::CompressedBufferWriter cbw;
  common::CompressedByteT* dst_data_d;
  common::CompressedIterator<uint32_t> src_iterator_d;
  /*! \brief An array that maps the rows from the full DMatrix to the compacted page.
   *
   * The total size is the number of rows in the original, uncompacted DMatrix. Elements are the
   * row ids in the compacted page. Rows not needed are set to SIZE_MAX.
   *
   * An example compacting 16 rows to 8 rows:
   * [SIZE_MAX, 0, 1, SIZE_MAX, SIZE_MAX, 2, SIZE_MAX, 3, 4, 5, SIZE_MAX, 6, SIZE_MAX, 7, SIZE_MAX,
   * SIZE_MAX]
   */
  common::Span<size_t> row_indexes;
  size_t base_rowid;
  size_t row_stride;

  CompactPage(EllpackPageImpl* dst, EllpackPageImpl* src, common::Span<size_t> row_indexes)
      : cbw{dst->matrix.info.NumSymbols()},
        dst_data_d{dst->gidx_buffer.data()},
        src_iterator_d{src->gidx_buffer.data(), src->matrix.info.NumSymbols()},
        row_indexes(row_indexes),
        base_rowid{src->matrix.base_rowid},
        row_stride{src->matrix.info.row_stride} {}

  __device__ void operator()(size_t row_id) {
    size_t src_row = base_rowid + row_id;
    size_t dst_row = row_indexes[src_row];
    if (dst_row == SIZE_MAX) return;
    size_t dst_offset = dst_row * row_stride;
    size_t src_offset = row_id * row_stride;
    for (size_t j = 0; j < row_stride; j++) {
      cbw.AtomicWriteSymbol(dst_data_d, src_iterator_d[src_offset + j], dst_offset + j);
    }
  }
};

// Compacts the data from the given EllpackPage into the current page.
void EllpackPageImpl::Compact(int device, EllpackPageImpl* page, common::Span<size_t> row_indexes) {
  monitor_.StartCuda("Compact");
  CHECK_EQ(matrix.info.row_stride, page->matrix.info.row_stride);
  CHECK_EQ(matrix.info.NumSymbols(), page->matrix.info.NumSymbols());
  CHECK_LE(page->matrix.base_rowid + page->matrix.n_rows, row_indexes.size());
  dh::LaunchN(device, page->matrix.n_rows, CompactPage(this, page, row_indexes));
  monitor_.StopCuda("Compact");
}

// Construct an EllpackInfo based on histogram cuts of features.
EllpackInfo::EllpackInfo(int device,
                         bool is_dense,
                         size_t row_stride,
                         const common::HistogramCuts& hmat,
                         dh::BulkAllocator* ba)
    : is_dense(is_dense), row_stride(row_stride), n_bins(hmat.Ptrs().back()) {

  ba->Allocate(device,
               &feature_segments, hmat.Ptrs().size(),
               &gidx_fvalue_map, hmat.Values().size(),
               &min_fvalue, hmat.MinValues().size());
  dh::CopyVectorToDeviceSpan(gidx_fvalue_map, hmat.Values());
  dh::CopyVectorToDeviceSpan(min_fvalue, hmat.MinValues());
  dh::CopyVectorToDeviceSpan(feature_segments, hmat.Ptrs());
}

// Initialize the EllpackInfo for this page.
void EllpackPageImpl::InitInfo(int device,
                               bool is_dense,
                               size_t row_stride,
                               const common::HistogramCuts& hmat) {
  matrix.info = EllpackInfo(device, is_dense, row_stride, hmat, &ba_);
}

// Initialize the buffer to stored compressed features.
void EllpackPageImpl::InitCompressedData(int device, size_t num_rows) {
  size_t num_symbols = matrix.info.NumSymbols();

  // Required buffer size for storing data matrix in ELLPack format.
  size_t compressed_size_bytes = common::CompressedBufferWriter::CalculateBufferSize(
      matrix.info.row_stride * num_rows, num_symbols);
  ba_.Allocate(device, &gidx_buffer, compressed_size_bytes);

  thrust::fill(dh::tbegin(gidx_buffer), dh::tend(gidx_buffer), 0);

  matrix.gidx_iter = common::CompressedIterator<uint32_t>(gidx_buffer.data(), num_symbols);
}

// Compress a CSR page into ELLPACK.
void EllpackPageImpl::CreateHistIndices(int device,
                                        const SparsePage& row_batch,
                                        const RowStateOnDevice& device_row_state) {
  // Has any been allocated for me in this batch?
  if (!device_row_state.rows_to_process_from_batch) return;

  unsigned int null_gidx_value = matrix.info.n_bins;
  size_t row_stride = matrix.info.row_stride;

  const auto& offset_vec = row_batch.offset.ConstHostVector();

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
    dh::device_vector<size_t> row_ptrs(batch_nrows + 1);
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
    dh::LaunchKernel {grid3, block3} (
        CompressBinEllpackKernel,
        common::CompressedBufferWriter(matrix.info.NumSymbols()),
        gidx_buffer.data(),
        row_ptrs.data().get(),
        entries_d.data().get(),
        matrix.info.gidx_fvalue_map.data(),
        matrix.info.feature_segments.data(),
        device_row_state.total_rows_processed + batch_row_begin,
        batch_nrows,
        row_stride,
        null_gidx_value);
  }
}

// Return the number of rows contained in this page.
size_t EllpackPageImpl::Size() const {
  return matrix.n_rows;
}

// Clear the current page.
void EllpackPageImpl::Clear() {
  ba_.Clear();
  gidx_buffer = {};
  idx_buffer.clear();
  sparse_page_.Clear();
  matrix.base_rowid = 0;
  matrix.n_rows = 0;
  device_initialized_ = false;
}

// Push a CSR page to the current page.
//
// The CSR pages are accumulated in memory until they reach a certain size, then written out as
// compressed ELLPACK.
void EllpackPageImpl::Push(int device, const SparsePage& batch) {
  sparse_page_.Push(batch);
  matrix.n_rows += batch.Size();
}

// Compress the accumulated SparsePage.
void EllpackPageImpl::CompressSparsePage(int device) {
  monitor_.StartCuda("InitCompressedData");
  InitCompressedData(device, matrix.n_rows);
  monitor_.StopCuda("InitCompressedData");

  monitor_.StartCuda("BinningCompression");
  DeviceHistogramBuilderState hist_builder_row_state(matrix.n_rows);
  hist_builder_row_state.BeginBatch(sparse_page_);
  CreateHistIndices(device, sparse_page_, hist_builder_row_state.GetRowStateOnDevice());
  hist_builder_row_state.EndBatch();
  monitor_.StopCuda("BinningCompression");

  monitor_.StartCuda("CopyDeviceToHost");
  idx_buffer.resize(gidx_buffer.size());
  dh::CopyDeviceSpanToVector(&idx_buffer, gidx_buffer);
  ba_.Clear();
  gidx_buffer = {};
  monitor_.StopCuda("CopyDeviceToHost");
}

// Return the memory cost for storing the compressed features.
size_t EllpackPageImpl::MemCostBytes() const {
  // Required buffer size for storing data matrix in ELLPack format.
  size_t compressed_size_bytes = common::CompressedBufferWriter::CalculateBufferSize(
      matrix.info.row_stride * matrix.n_rows, matrix.info.NumSymbols());
  return compressed_size_bytes;
}

// Copy the compressed features to GPU.
void EllpackPageImpl::InitDevice(int device, EllpackInfo info) {
  if (device_initialized_) return;

  monitor_.StartCuda("CopyPageToDevice");
  dh::safe_cuda(cudaSetDevice(device));

  gidx_buffer = {};
  ba_.Allocate(device, &gidx_buffer, idx_buffer.size());
  dh::CopyVectorToDeviceSpan(gidx_buffer, idx_buffer);

  matrix.info = info;
  matrix.gidx_iter = common::CompressedIterator<uint32_t>(gidx_buffer.data(), info.n_bins + 1);

  monitor_.StopCuda("CopyPageToDevice");

  device_initialized_ = true;
}
}  // namespace xgboost
