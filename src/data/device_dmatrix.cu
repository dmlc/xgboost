/*!
 * Copyright 2020 by Contributors
 * \file device_dmatrix.cu
 * \brief Device-memory version of DMatrix.
 */

#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <memory>
#include <utility>
#include "../common/hist_util.h"
#include "../common/math.h"
#include "adapter.h"
#include "device_adapter.cuh"
#include "ellpack_page.cuh"
#include "device_dmatrix.h"

namespace xgboost {
namespace data {

struct IsValidFunctor : public thrust::unary_function<Entry, bool> {
  explicit IsValidFunctor(float missing) : missing(missing) {}

  float missing;
  __device__ bool operator()(const data::COOTuple& e) const {
    if (common::CheckNAN(e.value) || e.value == missing) {
      return false;
    }
    return true;
  }
};

// Returns maximum row length
template <typename AdapterBatchT>
size_t GetRowCounts(const AdapterBatchT& batch, common::Span<size_t> offset,
                    int device_idx, float missing) {
  IsValidFunctor is_valid(missing);
  // Count elements per row
  dh::LaunchN(device_idx, batch.Size(), [=] __device__(size_t idx) {
    auto element = batch.GetElement(idx);
    if (is_valid(element)) {
      atomicAdd(reinterpret_cast<unsigned long long*>(  // NOLINT
                    &offset[element.row_idx]),
                static_cast<unsigned long long>(1));  // NOLINT
    }
  });
  dh::XGBCachingDeviceAllocator<char> alloc;
  size_t row_stride = thrust::reduce(
      thrust::cuda::par(alloc), thrust::device_pointer_cast(offset.data()),
      thrust::device_pointer_cast(offset.data()) + offset.size(), size_t(0),
      thrust::maximum<size_t>());
  return row_stride;
}

template <typename AdapterBatchT>
struct WriteCompressedEllpackFunctor {
  WriteCompressedEllpackFunctor(common::CompressedByteT* buffer,
                                const common::CompressedBufferWriter& writer,
                                const AdapterBatchT& batch,
                                EllpackDeviceAccessor accessor,
                                const IsValidFunctor& is_valid)
      : d_buffer(buffer),
        writer(writer),
        batch(batch),
        accessor(std::move(accessor)),
        is_valid(is_valid) {}

  common::CompressedByteT* d_buffer;
  common::CompressedBufferWriter writer;
  AdapterBatchT batch;
  EllpackDeviceAccessor accessor;
  IsValidFunctor is_valid;

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

// Here the data is already correctly ordered and simply needs to be compacted
// to remove missing data
template <typename AdapterBatchT>
void CopyDataRowMajor(const AdapterBatchT& batch, EllpackPageImpl* dst,
                      int device_idx, float missing,
                      common::Span<size_t> row_counts) {
  // Some witchcraft happens here
  // The goal is to copy valid elements out of the input to an ellpack matrix
  // with a given row stride, using no extra working memory Standard stream
  // compaction needs to be modified to do this, so we manually define a
  // segmented stream compaction via operators on an inclusive scan. The output
  // of this inclusive scan is fed to a custom function which works out the
  // correct output position
  auto counting = thrust::make_counting_iterator(0llu);
  IsValidFunctor is_valid(missing);
  auto key_iter = dh::MakeTransformIterator<size_t>(
      counting,
      [=] __device__(size_t idx) { return batch.GetElement(idx).row_idx; });
  auto value_iter = dh::MakeTransformIterator<size_t>(
      counting, [=] __device__(size_t idx) -> size_t {
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
  thrust::discard_iterator<size_t> discard;
  thrust::transform_output_iterator<
      WriteCompressedEllpackFunctor<AdapterBatchT>, decltype(discard)>
      out(discard, functor);
  dh::XGBCachingDeviceAllocator<char> alloc;
  thrust::inclusive_scan(thrust::cuda::par(alloc), key_value_index_iter,
                         key_value_index_iter + batch.Size(), out,
                         [=] __device__(Tuple a, Tuple b) {
                           // Key equal
                           if (a.get<0>() == b.get<0>()) {
                             b.get<1>() += a.get<1>();
                             return b;
                           }
                           // Not equal
                           return b;
                         });
}

template <typename AdapterT, typename AdapterBatchT>
void CopyDataColumnMajor(AdapterT* adapter, const AdapterBatchT& batch,
                         EllpackPageImpl* dst, float missing) {
  // Step 1: Get the sizes of the input columns
  dh::caching_device_vector<size_t> column_sizes(adapter->NumColumns(), 0);
  auto d_column_sizes = column_sizes.data().get();
  // Populate column sizes
  dh::LaunchN(adapter->DeviceIdx(), batch.Size(), [=] __device__(size_t idx) {
    const auto& e = batch.GetElement(idx);
    atomicAdd(reinterpret_cast<unsigned long long*>(  // NOLINT
                  &d_column_sizes[e.column_idx]),
              static_cast<unsigned long long>(1));  // NOLINT
  });

  thrust::host_vector<size_t> host_column_sizes = column_sizes;

  // Step 2: Iterate over columns, place elements in correct row, increment
  // temporary row pointers
  dh::caching_device_vector<size_t> temp_row_ptr(adapter->NumRows(), 0);
  auto d_temp_row_ptr = temp_row_ptr.data().get();
  auto row_stride = dst->row_stride;
  size_t begin = 0;
  auto device_accessor = dst->GetDeviceAccessor(adapter->DeviceIdx());
  common::CompressedBufferWriter writer(device_accessor.NumSymbols());
  auto d_compressed_buffer = dst->gidx_buffer.DevicePointer();
  IsValidFunctor is_valid(missing);
  for (auto size : host_column_sizes) {
    size_t end = begin + size;
    dh::LaunchN(adapter->DeviceIdx(), end - begin, [=] __device__(size_t idx) {
      auto writer_non_const =
          writer;  // For some reason this variable gets captured as const
      const auto& e = batch.GetElement(idx + begin);
      if (!is_valid(e)) return;
      size_t output_position =
          e.row_idx * row_stride + d_temp_row_ptr[e.row_idx];
      auto bin_idx = device_accessor.SearchBin(e.value, e.column_idx);
      writer_non_const.AtomicWriteSymbol(d_compressed_buffer, bin_idx,
                                         output_position);
      d_temp_row_ptr[e.row_idx] += 1;
    });

    begin = end;
  }
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
// Does not currently support metainfo as no on-device data source contains this
// Current implementation assumes a single batch. More batches can
// be supported in future. Does not currently support inferring row/column size
template <typename AdapterT>
DeviceDMatrix::DeviceDMatrix(AdapterT* adapter, float missing, int nthread) {
  common::HistogramCuts cuts =
      common::AdapterDeviceSketch(adapter, 256, missing);
  auto& batch = adapter->Value();
  // Work out how many valid entries we have in each row
  dh::caching_device_vector<size_t> row_counts(adapter->NumRows() + 1, 0);
  common::Span<size_t> row_counts_span(row_counts.data().get(),
                                       row_counts.size());
  size_t row_stride =
      GetRowCounts(batch, row_counts_span, adapter->DeviceIdx(), missing);

  dh::XGBCachingDeviceAllocator<char> alloc;
  info.num_nonzero_ = thrust::reduce(thrust::cuda::par(alloc),
                                     row_counts.begin(), row_counts.end());
  info.num_col_ = adapter->NumColumns();
  info.num_row_ = adapter->NumRows();
  ellpack_page_.reset(new EllpackPage());
  *ellpack_page_->Impl() =
      EllpackPageImpl(adapter->DeviceIdx(), cuts, this->IsDense(), row_stride,
                      adapter->NumRows());
  if (adapter->IsRowMajor()) {
    CopyDataRowMajor(batch, ellpack_page_->Impl(), adapter->DeviceIdx(),
                     missing, row_counts_span);
  } else {
    CopyDataColumnMajor(adapter, batch, ellpack_page_->Impl(), missing);
  }

  WriteNullValues(ellpack_page_->Impl(), adapter->DeviceIdx(), row_counts_span);

  // Synchronise worker columns
  rabit::Allreduce<rabit::op::Max>(&info.num_col_, 1);
}
template DeviceDMatrix::DeviceDMatrix(CudfAdapter* adapter, float missing,
                                      int nthread);
template DeviceDMatrix::DeviceDMatrix(CupyAdapter* adapter, float missing,
                                      int nthread);
}  // namespace data
}  // namespace xgboost
