/*!
 * Copyright 2020 by Contributors
 * \file device_dmatrix.cu
 * \brief Device-memory version of DMatrix.
 */

#include <xgboost/base.h>
#include <xgboost/data.h>

#include <memory>
#include <thrust/execution_policy.h>

#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include "adapter.h"
#include "simple_dmatrix.h"
#include "device_dmatrix.h"
#include "device_adapter.cuh"
#include "ellpack_page.cuh"
#include "../common/hist_util.h"
#include "../common/math.h"

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
size_t CountRowOffsets(const AdapterBatchT& batch, common::Span<size_t> offset,
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
      thrust::device_pointer_cast(offset.data()) + offset.size(),
                     size_t(0),
                     thrust::maximum<size_t>());

  thrust::exclusive_scan(thrust::cuda::par(alloc),
      thrust::device_pointer_cast(offset.data()),
      thrust::device_pointer_cast(offset.data() + offset.size()),
      thrust::device_pointer_cast(offset.data()));
  return row_stride;
}
template <typename FunctionT>
class LauncherItr {
public:
  int idx;
  XGBOOST_DEVICE LauncherItr() : idx(0) {}
  XGBOOST_DEVICE LauncherItr(int idx) : idx(idx){}
  XGBOOST_DEVICE LauncherItr &operator=(int output) {
    FunctionT f;
    f(idx, output);
    return *this;
  }
};

/**
* \brief Thrust compatible iterator type - discards algorithm output and launches device lambda
*        with the index of the output and the algorithm output as arguments.
*
* \author  Rory
* \date  7/9/2017
*
* \tparam  FunctionT Type of the function t.
*/
template <typename FunctionT>
class DiscardLambdaItr {
public:
  // Required iterator traits
  using self_type = DiscardLambdaItr<FunctionT>;  // NOLINT
  using difference_type = ptrdiff_t;   // NOLINT
  using value_type = void;       // NOLINT
  using pointer = value_type *;  // NOLINT
  using reference = LauncherItr<FunctionT>;  // NOLINT
  using iterator_category = typename thrust::detail::iterator_facade_category<
    thrust::any_system_tag, thrust::random_access_traversal_tag, value_type,
    reference>::type;  // NOLINT
private:
  difference_type offset_;
public:
  XGBOOST_DEVICE explicit DiscardLambdaItr() : offset_(0){}
  XGBOOST_DEVICE explicit DiscardLambdaItr(size_t offset) : offset_(offset){}
  XGBOOST_DEVICE self_type operator+(const int &b) const {
    return self_type(offset_ + b);
  }
  XGBOOST_DEVICE self_type operator++() {
    offset_++;
    return *this;
  }
  XGBOOST_DEVICE self_type operator++(int) {
    self_type retval = *this;
    offset_++;
    return retval;
  }
  XGBOOST_DEVICE self_type &operator+=(const int &b) {
    offset_ += b;
    return *this;
  }
  XGBOOST_DEVICE reference operator*() const {
    return LauncherItr<FunctionT>(offset_);
  }
  XGBOOST_DEVICE reference operator[](int idx) {
    self_type offset = (*this) + idx;
    return *offset;
  }
};

  template <typename AdapterBatchT>
struct WriteCompressedEllpackFunctor
{
  WriteCompressedEllpackFunctor(common::CompressedByteT* buffer,
    const common::CompressedBufferWriter& writer, const AdapterBatchT& batch,const EllpackDeviceAccessor& accessor,const IsValidFunctor&is_valid)
    :
      d_buffer(buffer),
      writer(writer),
      batch(batch),accessor(accessor),is_valid(is_valid)
  {
  }

  common::CompressedByteT* d_buffer;
  common::CompressedBufferWriter writer;
  AdapterBatchT batch;
  EllpackDeviceAccessor accessor;
  IsValidFunctor is_valid;

  using Tuple = thrust::tuple<size_t, size_t, size_t>;
  __device__ size_t operator()(Tuple  out)
  {
    auto e = batch.GetElement(out.get<2>());
    if (is_valid(e)) {
      size_t output_position = accessor.row_stride * e.row_idx + out.get<1>() - 1;
      auto bin_idx = accessor.SearchBin(e.value, e.column_idx);
      writer.AtomicWriteSymbol(d_buffer, bin_idx, output_position);
    }
    return 0;
    
  }
};

// Here the data is already correctly ordered and simply needs to be compacted
// to remove missing data
template <typename AdapterBatchT>
void CopyDataRowMajor(const AdapterBatchT& batch, EllpackPageImpl*dst,
                      int device_idx, float missing) {
  auto counting = thrust::make_counting_iterator(0llu);
  IsValidFunctor is_valid(missing);
  auto key_iter = dh::MakeTransformIterator<size_t >(counting,[=]__device__ (size_t idx)
  {
    return batch.GetElement(idx).row_idx;
  });
  auto value_iter = dh::MakeTransformIterator<size_t>(
      counting,
      [=]__device__ (size_t idx) -> size_t 
  {
    return is_valid(batch.GetElement(idx));
  });

  auto key_value_index_iter = thrust::make_zip_iterator(thrust::make_tuple(key_iter, value_iter, counting));
  using Tuple = thrust::tuple<size_t , size_t , size_t >;

  auto device_accessor = dst->GetDeviceAccessor(device_idx);
  common::CompressedBufferWriter writer(device_accessor.NumSymbols());
  auto d_compressed_buffer = dst->gidx_buffer.DevicePointer();

  WriteCompressedEllpackFunctor<AdapterBatchT> functor(d_compressed_buffer, writer,
                                        batch, device_accessor, is_valid);
  thrust::discard_iterator<size_t> discard;
  thrust::transform_output_iterator<
      WriteCompressedEllpackFunctor<AdapterBatchT>, decltype(discard)>
      out(discard, functor);
  dh::XGBCachingDeviceAllocator<char> alloc;
  thrust::inclusive_scan(
      thrust::cuda::par(alloc), key_value_index_iter, key_value_index_iter + batch.Size(), out,
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

//void RedirectScan() {
//  int n = 5;
//  thrust::device_vector<int> x(n, 2);
//  auto counting = thrust::make_counting_iterator(0llu);
//  using Tuple = thrust::tuple<size_t , size_t >;
//  auto zip = thrust::make_zip_iterator(thrust::make_tuple(x.begin(), counting));
//  thrust::discard_iterator<int > discard;
//  thrust::transform_output_iterator<Functor,decltype(discard)> out(discard,Functor());
//  thrust::inclusive_scan(thrust::device, zip, zip + x.size(), out,[=]__device__ (Tuple a ,Tuple b)
//  {
//    b.get<0>() +=a.get<0>();
//    //b.get<1>() = b.get<1>() + 1;
//    return b;
//  });
//}
//
// Does not currently support metainfo as no on-device data source contains this
// Current implementation assumes a single batch. More batches can
// be supported in future. Does not currently support inferring row/column size
  template <typename AdapterT>
DeviceDMatrix::DeviceDMatrix(AdapterT* adapter, float missing, int nthread) {
  //RedirectScan();
  common::HistogramCuts cuts = common::AdapterDeviceSketch(adapter, 256, missing);
  auto & batch = adapter->Value();
  // Work out how many valid entries we have in each row
  dh::caching_device_vector<size_t> row_ptr(adapter->NumRows() + 1,
                                                      0);
  common::Span<size_t > row_ptr_span( row_ptr.data().get(),row_ptr.size() );
  // TODO: are these offsets actually needed? or just the stride
  size_t row_stride =
      CountRowOffsets(batch, row_ptr_span, adapter->DeviceIdx(), missing);

  info.num_nonzero_ = row_ptr.back();// Device to host copy
  info.num_col_ = adapter->NumColumns();
  info.num_row_ = adapter->NumRows();
  ellpack_page_.reset(new EllpackPage());
  *ellpack_page_->Impl() =
      EllpackPageImpl(adapter->DeviceIdx(), cuts, this->IsDense(), row_stride,
                      adapter->NumRows());
  if (adapter->IsRowMajor()) {
    CopyDataRowMajor(batch, ellpack_page_->Impl(), adapter->DeviceIdx(), missing);
  }

  // Synchronise worker columns
  rabit::Allreduce<rabit::op::Max>(&info.num_col_, 1);
}
template DeviceDMatrix::DeviceDMatrix(CudfAdapter* adapter, float missing,
                                      int nthread);
template DeviceDMatrix::DeviceDMatrix(CupyAdapter* adapter, float missing,
                                      int nthread);
}  // namespace data
}  // namespace xgboost
