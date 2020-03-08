/*!
 * Copyright 2019 XGBoost contributors
 */

#include <xgboost/data.h>

#include "../common/hist_util.h"
#include "../common/random.h"
#include "./ellpack_page.cuh"

namespace xgboost {

EllpackPage::EllpackPage() : impl_{new EllpackPageImpl()} {}

EllpackPage::EllpackPage(DMatrix* dmat, const BatchParam& param)
    : impl_{new EllpackPageImpl(dmat, param)} {}

EllpackPage::~EllpackPage() = default;

size_t EllpackPage::Size() const { return impl_->Size(); }

void EllpackPage::SetBaseRowId(size_t row_id) { impl_->SetBaseRowId(row_id); }

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
EllpackPageImpl::EllpackPageImpl(int device, common::HistogramCuts cuts,
                                 bool is_dense, size_t row_stride,
                                 size_t n_rows)
    : is_dense(is_dense),
      cuts_(std::move(cuts)),
      row_stride(row_stride),
      n_rows(n_rows) {
  monitor_.Init("ellpack_page");
  dh::safe_cuda(cudaSetDevice(device));

  monitor_.StartCuda("InitCompressedData");
  InitCompressedData(device);
  monitor_.StopCuda("InitCompressedData");
  // InitDevice();
}

size_t GetRowStride(DMatrix* dmat) {
  if (dmat->IsDense()) return dmat->Info().num_col_;

  size_t row_stride = 0;
  for (const auto& batch : dmat->GetBatches<SparsePage>()) {
    const auto& row_offset = batch.offset.ConstHostVector();
    for (auto i = 1ull; i < row_offset.size(); i++) {
      row_stride = std::max(
        row_stride, static_cast<size_t>(row_offset[i] - row_offset[i - 1]));
    }
  }
  return row_stride;
}

// Construct an ELLPACK matrix in memory.
EllpackPageImpl::EllpackPageImpl(DMatrix* dmat, const BatchParam& param)
    : is_dense(dmat->IsDense()) {
  monitor_.Init("ellpack_page");
  dh::safe_cuda(cudaSetDevice(param.gpu_id));

  n_rows = dmat->Info().num_row_;

  monitor_.StartCuda("Quantiles");
  // Create the quantile sketches for the dmatrix and initialize HistogramCuts.
  row_stride = GetRowStride(dmat);
<<<<<<< HEAD
  cuts_ = common::DeviceSketch(param.gpu_id, dmat, param.max_bin);
=======
  cuts_ = common::DeviceSketch(param.gpu_id, dmat, param.max_bin,
                                   param.gpu_batch_nrows);
>>>>>>> Rebase
  monitor_.StopCuda("Quantiles");

  monitor_.StartCuda("InitCompressedData");
  InitCompressedData(param.gpu_id);
  monitor_.StopCuda("InitCompressedData");

  monitor_.StartCuda("BinningCompression");
  for (const auto& batch : dmat->GetBatches<SparsePage>()) {
    CreateHistIndices(param.gpu_id, batch);
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
      : cbw{dst->NumSymbols()},
        dst_data_d{dst->gidx_buffer.DevicePointer()},
        src_iterator_d{src->gidx_buffer.DevicePointer(), src->NumSymbols()},
        offset(offset) {}

  __device__ void operator()(size_t element_id) {
    cbw.AtomicWriteSymbol(dst_data_d, src_iterator_d[element_id],
                          element_id + offset);
  }
};

// Copy the data from the given EllpackPage to the current page.
size_t EllpackPageImpl::Copy(int device, EllpackPageImpl* page, size_t offset) {
  monitor_.StartCuda("Copy");
  size_t num_elements = page->n_rows * page->row_stride;
  CHECK_EQ(row_stride, page->row_stride);
  CHECK_EQ(NumSymbols(), page->NumSymbols());
  CHECK_GE(n_rows * row_stride, offset + num_elements);
  gidx_buffer.SetDevice(device);
  page->gidx_buffer.SetDevice(device);
  dh::LaunchN(device, num_elements, CopyPage(this, page, offset));
  monitor_.StopCuda("Copy");
  return num_elements;
}

// A functor that compacts the rows from one EllpackPage into another.
struct CompactPage {
  common::CompressedBufferWriter cbw;
  common::CompressedByteT* dst_data_d;
  common::CompressedIterator<uint32_t> src_iterator_d;
  /*! \brief An array that maps the rows from the full DMatrix to the compacted
   * page.
   *
   * The total size is the number of rows in the original, uncompacted DMatrix.
   * Elements are the row ids in the compacted page. Rows not needed are set to
   * SIZE_MAX.
   *
   * An example compacting 16 rows to 8 rows:
   * [SIZE_MAX, 0, 1, SIZE_MAX, SIZE_MAX, 2, SIZE_MAX, 3, 4, 5, SIZE_MAX, 6,
   * SIZE_MAX, 7, SIZE_MAX, SIZE_MAX]
   */
  common::Span<size_t> row_indexes;
  size_t base_rowid;
  size_t row_stride;

  CompactPage(EllpackPageImpl* dst, EllpackPageImpl* src,
              common::Span<size_t> row_indexes)
      : cbw{dst->NumSymbols()},
        dst_data_d{dst->gidx_buffer.DevicePointer()},
        src_iterator_d{src->gidx_buffer.DevicePointer(), src->NumSymbols()},
        row_indexes(row_indexes),
        base_rowid{src->base_rowid},
        row_stride{src->row_stride} {}

  __device__ void operator()(size_t row_id) {
    size_t src_row = base_rowid + row_id;
    size_t dst_row = row_indexes[src_row];
    if (dst_row == SIZE_MAX) return;
    size_t dst_offset = dst_row * row_stride;
    size_t src_offset = row_id * row_stride;
    for (size_t j = 0; j < row_stride; j++) {
      cbw.AtomicWriteSymbol(dst_data_d, src_iterator_d[src_offset + j],
                            dst_offset + j);
    }
  }
};

// Compacts the data from the given EllpackPage into the current page.
void EllpackPageImpl::Compact(int device, EllpackPageImpl* page,
                              common::Span<size_t> row_indexes) {
  monitor_.StartCuda("Compact");
  CHECK_EQ(row_stride, page->row_stride);
  CHECK_EQ(NumSymbols(), page->NumSymbols());
  CHECK_LE(page->base_rowid + page->n_rows, row_indexes.size());
  gidx_buffer.SetDevice(device);
  page->gidx_buffer.SetDevice(device);
  dh::LaunchN(device, page->n_rows, CompactPage(this, page, row_indexes));
  monitor_.StopCuda("Compact");
}

// Initialize the buffer to stored compressed features.
void EllpackPageImpl::InitCompressedData(int device) {
  size_t num_symbols = NumSymbols();

  // Required buffer size for storing data matrix in ELLPack format.
  size_t compressed_size_bytes =
      common::CompressedBufferWriter::CalculateBufferSize(row_stride * n_rows,
                                                          num_symbols);
  gidx_buffer.SetDevice(device);
  // Don't call fill unnecessarily
  if (gidx_buffer.Size() == 0) {
    gidx_buffer.Resize(compressed_size_bytes, 0);
  } else {
    gidx_buffer.Resize(compressed_size_bytes, 0);
    thrust::fill(dh::tbegin(gidx_buffer), dh::tend(gidx_buffer), 0);
  }
}

// Compress a CSR page into ELLPACK.
void EllpackPageImpl::CreateHistIndices(int device,
                                        const SparsePage& row_batch) {
  if (row_batch.Size() == 0) return;
  unsigned int null_gidx_value = NumSymbols() - 1;

  const auto& offset_vec = row_batch.offset.ConstHostVector();

  // bin and compress entries in batches of rows
  size_t gpu_batch_nrows =
      std::min(dh::TotalMemory(device) / (16 * row_stride * sizeof(Entry)),
               static_cast<size_t>(row_batch.Size()));
  const std::vector<Entry>& data_vec = row_batch.data.ConstHostVector();

  size_t gpu_nbatches = common::DivRoundUp(row_batch.Size(), gpu_batch_nrows);

  for (size_t gpu_batch = 0; gpu_batch < gpu_nbatches; ++gpu_batch) {
    size_t batch_row_begin = gpu_batch * gpu_batch_nrows;
    size_t batch_row_end =
        std::min((gpu_batch + 1) * gpu_batch_nrows, row_batch.Size());
    size_t batch_nrows = batch_row_end - batch_row_begin;

    const auto ent_cnt_begin = offset_vec[batch_row_begin];
    const auto ent_cnt_end = offset_vec[batch_row_end];

    /*! \brief row offset in SparsePage (the input data). */
    dh::device_vector<size_t> row_ptrs(batch_nrows + 1);
    thrust::copy(offset_vec.data() + batch_row_begin,
                 offset_vec.data() + batch_row_end + 1, row_ptrs.begin());

    // number of entries in this batch.
    size_t n_entries = ent_cnt_end - ent_cnt_begin;
    dh::device_vector<Entry> entries_d(n_entries);
    // copy data entries to device.
    dh::safe_cuda(cudaMemcpy(entries_d.data().get(),
                             data_vec.data() + ent_cnt_begin,
                             n_entries * sizeof(Entry), cudaMemcpyDefault));
    const dim3 block3(32, 8, 1);  // 256 threads
    const dim3 grid3(common::DivRoundUp(batch_nrows, block3.x),
                     common::DivRoundUp(row_stride, block3.y), 1);
    auto device_accessor = GetDeviceAccessor(device);
    dh::LaunchKernel {grid3, block3}(
        CompressBinEllpackKernel, common::CompressedBufferWriter(NumSymbols()),
        gidx_buffer.DevicePointer(), row_ptrs.data().get(),
        entries_d.data().get(), device_accessor.gidx_fvalue_map.data(),
        device_accessor.feature_segments.data(),
        row_batch.base_rowid + batch_row_begin, batch_nrows, row_stride,
        null_gidx_value);
  }
}

// Return the number of rows contained in this page.
size_t EllpackPageImpl::Size() const { return n_rows; }

// Return the memory cost for storing the compressed features.
size_t EllpackPageImpl::MemCostBytes(size_t num_rows, size_t row_stride,
                                     const common::HistogramCuts& cuts) {
  // Required buffer size for storing data matrix in EtoLLPack format.
  size_t compressed_size_bytes =
      common::CompressedBufferWriter::CalculateBufferSize(row_stride * num_rows,
                                                          cuts.TotalBins() + 1);
  return compressed_size_bytes;
}

EllpackDeviceAccessor EllpackPageImpl::GetDeviceAccessor(int device) const {
  gidx_buffer.SetDevice(device);
<<<<<<< HEAD
  return EllpackDeviceAccessor(
      device, cuts_, is_dense, row_stride, base_rowid, n_rows,
      common::CompressedIterator<uint32_t>(gidx_buffer.ConstDevicePointer(),
                                           NumSymbols()));
}

EllpackPageImpl::EllpackPageImpl(int device, common::HistogramCuts cuts,
                                 const SparsePage& page, bool is_dense,
                                 size_t row_stride)
    : cuts_(std::move(cuts)),
=======
  return EllpackDeviceAccessor(device, cuts_, is_dense, row_stride, base_rowid,
                               n_rows,
                               common::CompressedIterator<uint32_t>(
                                   gidx_buffer.ConstDevicePointer(), NumSymbols()));
}

EllpackPageImpl::EllpackPageImpl(int device, const common::HistogramCuts& cuts,
                                 const SparsePage& page, bool is_dense,
                                 size_t row_stride)
    : cuts_(cuts),
>>>>>>> Rebase
      is_dense(is_dense),
      n_rows(page.Size()),
      row_stride(row_stride) {
  this->InitCompressedData(device);
  this->CreateHistIndices(device, page);
}
}  // namespace xgboost
