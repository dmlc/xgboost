/*!
 * Copyright 2019-2020 XGBoost contributors
 */
#include <xgboost/data.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include "../common/categorical.h"
#include "../common/hist_util.cuh"
#include "../common/random.h"
#include "./ellpack_page.cuh"
#include "device_adapter.cuh"

namespace xgboost {

EllpackPage::EllpackPage() : impl_{new EllpackPageImpl()} {}

EllpackPage::EllpackPage(DMatrix* dmat, const BatchParam& param)
    : impl_{new EllpackPageImpl(dmat, param)} {}

EllpackPage::~EllpackPage() = default;

EllpackPage::EllpackPage(EllpackPage&& that) { std::swap(impl_, that.impl_); }

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
    common::Span<FeatureType const> feature_types,
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
    bool is_cat = common::IsCat(feature_types, ifeature);
    // Assigning the bin in current entry.
    // S.t.: fvalue < feature_cuts[bin]
    if (is_cat) {
      auto it = dh::MakeTransformIterator<int>(
          feature_cuts, [](float v) { return common::AsCat(v); });
      bin = thrust::lower_bound(thrust::seq, it, it + ncuts, common::AsCat(fvalue)) - it;
    } else {
      bin = thrust::upper_bound(thrust::seq, feature_cuts, feature_cuts + ncuts,
                                fvalue) -
            feature_cuts;
    }

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

  monitor_.Start("InitCompressedData");
  InitCompressedData(device);
  monitor_.Stop("InitCompressedData");
}

EllpackPageImpl::EllpackPageImpl(int device, common::HistogramCuts cuts,
                                 const SparsePage &page, bool is_dense,
                                 size_t row_stride,
                                 common::Span<FeatureType const> feature_types)
    : cuts_(std::move(cuts)), is_dense(is_dense), n_rows(page.Size()),
      row_stride(row_stride) {
  this->InitCompressedData(device);
  this->CreateHistIndices(device, page, feature_types);
}

// Construct an ELLPACK matrix in memory.
EllpackPageImpl::EllpackPageImpl(DMatrix* dmat, const BatchParam& param)
    : is_dense(dmat->IsDense()) {
  monitor_.Init("ellpack_page");
  dh::safe_cuda(cudaSetDevice(param.gpu_id));

  n_rows = dmat->Info().num_row_;

  monitor_.Start("Quantiles");
  // Create the quantile sketches for the dmatrix and initialize HistogramCuts.
  row_stride = GetRowStride(dmat);
  cuts_ = common::DeviceSketch(param.gpu_id, dmat, param.max_bin);
  monitor_.Stop("Quantiles");

  monitor_.Start("InitCompressedData");
  this->InitCompressedData(param.gpu_id);
  monitor_.Stop("InitCompressedData");

  dmat->Info().feature_types.SetDevice(param.gpu_id);
  auto ft = dmat->Info().feature_types.ConstDeviceSpan();
  monitor_.Start("BinningCompression");
  for (const auto& batch : dmat->GetBatches<SparsePage>()) {
    CreateHistIndices(param.gpu_id, batch, ft);
  }
  monitor_.Stop("BinningCompression");
}

template <typename AdapterBatchT>
struct WriteCompressedEllpackFunctor {
  WriteCompressedEllpackFunctor(common::CompressedByteT* buffer,
                                const common::CompressedBufferWriter& writer,
                                AdapterBatchT batch,
                                EllpackDeviceAccessor accessor,
                                const data::IsValidFunctor& is_valid)
      : d_buffer(buffer),
      writer(writer),
      batch(std::move(batch)),
      accessor(std::move(accessor)),
      is_valid(is_valid) {}

  common::CompressedByteT* d_buffer;
  common::CompressedBufferWriter writer;
  AdapterBatchT batch;
  EllpackDeviceAccessor accessor;
  data::IsValidFunctor is_valid;

  using Tuple = thrust::tuple<size_t, size_t, size_t>;
  __device__ size_t operator()(Tuple out) {
    auto e = batch.GetElement(out.get<2>());
    if (is_valid(e)) {
      // -1 because the scan is inclusive
      size_t output_position =
          accessor.row_stride * e.row_idx + out.get<1>() - 1;
      auto bin_idx = accessor.SearchBin(e.value, e.column_idx);
      writer.AtomicWriteSymbol(d_buffer, bin_idx, output_position);
    }
    return 0;
  }
};

template <typename Tuple>
struct TupleScanOp {
  __device__ Tuple operator()(Tuple a, Tuple b) {
    // Key equal
    if (a.template get<0>() == b.template get<0>()) {
      b.template get<1>() += a.template get<1>();
      return b;
    }
    // Not equal
    return b;
  }
};

// Change the value type of thrust discard iterator so we can use it with cub
template <typename T>
class TypedDiscard : public thrust::discard_iterator<T> {
 public:
  using value_type = T;  // NOLINT
};

// Here the data is already correctly ordered and simply needs to be compacted
// to remove missing data
template <typename AdapterBatchT>
void CopyDataToEllpack(const AdapterBatchT& batch, EllpackPageImpl* dst,
                       int device_idx, float missing) {
  // Some witchcraft happens here
  // The goal is to copy valid elements out of the input to an ellpack matrix
  // with a given row stride, using no extra working memory Standard stream
  // compaction needs to be modified to do this, so we manually define a
  // segmented stream compaction via operators on an inclusive scan. The output
  // of this inclusive scan is fed to a custom function which works out the
  // correct output position
  auto counting = thrust::make_counting_iterator(0llu);
  data::IsValidFunctor is_valid(missing);
  auto key_iter = dh::MakeTransformIterator<size_t>(
      counting,
      [=] __device__(size_t idx) {
        return batch.GetElement(idx).row_idx;
      });
  auto value_iter = dh::MakeTransformIterator<size_t>(
      counting,
      [=] __device__(size_t idx) -> size_t {
        return is_valid(batch.GetElement(idx));
      });

  auto key_value_index_iter = thrust::make_zip_iterator(
      thrust::make_tuple(key_iter, value_iter, counting));

  // Tuple[0] = The row index of the input, used as a key to define segments
  // Tuple[1] = Scanned flags of valid elements for each row
  // Tuple[2] = The index in the input data
  using Tuple = thrust::tuple<size_t, size_t, size_t>;

  auto device_accessor = dst->GetDeviceAccessor(device_idx);
  common::CompressedBufferWriter writer(device_accessor.NumSymbols());
  auto d_compressed_buffer = dst->gidx_buffer.DevicePointer();

  // We redirect the scan output into this functor to do the actual writing
  WriteCompressedEllpackFunctor<AdapterBatchT> functor(
      d_compressed_buffer, writer, batch, device_accessor, is_valid);
  TypedDiscard<Tuple> discard;
  thrust::transform_output_iterator<
    WriteCompressedEllpackFunctor<AdapterBatchT>, decltype(discard)>
      out(discard, functor);
  // Go one level down into cub::DeviceScan API to set OffsetT as 64 bit
  // So we don't crash on n > 2^31
  size_t temp_storage_bytes = 0;
  using DispatchScan =
      cub::DispatchScan<decltype(key_value_index_iter), decltype(out),
                        TupleScanOp<Tuple>, cub::NullType, int64_t>;
  DispatchScan::Dispatch(nullptr, temp_storage_bytes, key_value_index_iter, out,
                         TupleScanOp<Tuple>(), cub::NullType(), batch.Size(),
                         nullptr, false);
  dh::TemporaryArray<char> temp_storage(temp_storage_bytes);
  DispatchScan::Dispatch(temp_storage.data().get(), temp_storage_bytes,
                         key_value_index_iter, out, TupleScanOp<Tuple>(),
                         cub::NullType(), batch.Size(), nullptr, false);
}

void WriteNullValues(EllpackPageImpl* dst, int device_idx,
                     common::Span<size_t> row_counts) {
  // Write the null values
  auto device_accessor = dst->GetDeviceAccessor(device_idx);
  common::CompressedBufferWriter writer(device_accessor.NumSymbols());
  auto d_compressed_buffer = dst->gidx_buffer.DevicePointer();
  auto row_stride = dst->row_stride;
  dh::LaunchN(device_idx, row_stride * dst->n_rows, [=] __device__(size_t idx) {
      auto writer_non_const =
          writer;  // For some reason this variable gets captured as const
      size_t row_idx = idx / row_stride;
      size_t row_offset = idx % row_stride;
      if (row_offset >= row_counts[row_idx]) {
        writer_non_const.AtomicWriteSymbol(d_compressed_buffer,
                                           device_accessor.NullValue(), idx);
      }
    });
}

template <typename AdapterBatch>
EllpackPageImpl::EllpackPageImpl(AdapterBatch batch, float missing, int device,
                                 bool is_dense, int nthread,
                                 common::Span<size_t> row_counts_span,
                                 size_t row_stride, size_t n_rows, size_t n_cols,
                                 common::HistogramCuts const& cuts) {
  dh::safe_cuda(cudaSetDevice(device));

  *this = EllpackPageImpl(device, cuts, is_dense, row_stride, n_rows);
  CopyDataToEllpack(batch, this, device, missing);
  WriteNullValues(this, device, row_counts_span);
}

#define ELLPACK_BATCH_SPECIALIZE(__BATCH_T)                             \
  template EllpackPageImpl::EllpackPageImpl(                            \
      __BATCH_T batch, float missing, int device,                       \
      bool is_dense, int nthread,                                       \
      common::Span<size_t> row_counts_span,                             \
      size_t row_stride, size_t n_rows, size_t n_cols,                  \
      common::HistogramCuts const& cuts);

ELLPACK_BATCH_SPECIALIZE(data::CudfAdapterBatch)
ELLPACK_BATCH_SPECIALIZE(data::CupyAdapterBatch)

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
  monitor_.Start("Copy");
  size_t num_elements = page->n_rows * page->row_stride;
  CHECK_EQ(row_stride, page->row_stride);
  CHECK_EQ(NumSymbols(), page->NumSymbols());
  CHECK_GE(n_rows * row_stride, offset + num_elements);
  if (page == this) {
    LOG(FATAL) << "Concatenating the same Ellpack.";
    return this->n_rows * this->row_stride;
  }
  gidx_buffer.SetDevice(device);
  page->gidx_buffer.SetDevice(device);
  dh::LaunchN(device, num_elements, CopyPage(this, page, offset));
  monitor_.Stop("Copy");
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
  monitor_.Start("Compact");
  CHECK_EQ(row_stride, page->row_stride);
  CHECK_EQ(NumSymbols(), page->NumSymbols());
  CHECK_LE(page->base_rowid + page->n_rows, row_indexes.size());
  gidx_buffer.SetDevice(device);
  page->gidx_buffer.SetDevice(device);
  dh::LaunchN(device, page->n_rows, CompactPage(this, page, row_indexes));
  monitor_.Stop("Compact");
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
                                        const SparsePage& row_batch,
                                        common::Span<FeatureType const> feature_types) {
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
    dh::safe_cuda(cudaMemcpyAsync(entries_d.data().get(),
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
        device_accessor.feature_segments.data(), feature_types,
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
  return EllpackDeviceAccessor(
      device, cuts_, is_dense, row_stride, base_rowid, n_rows,
      common::CompressedIterator<uint32_t>(gidx_buffer.ConstDevicePointer(),
                                           NumSymbols()));
}
}  // namespace xgboost
