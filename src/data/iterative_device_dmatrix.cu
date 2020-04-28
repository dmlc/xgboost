/*!
 * Copyright 2020 XGBoost contributors
 */
#include <memory>
#include <type_traits>

#include "../common/quantile.cuh"
#include "simple_batch_iterator.h"
#include "iterative_device_dmatrix.h"
#include "sparse_page_source.h"
#include "ellpack_page.cuh"
#include "proxy_dmatrix.h"
#include "device_adapter.cuh"

namespace xgboost {
namespace data {

#define DISPATCH_MEM(__Proxy, __Fn)                                     \
  [](DMatrixProxy const* proxy) -> decltype(                            \
      (dmlc::get<CupyAdapter>(proxy->Adapter()).Value()).__Fn()) {      \
    CHECK(proxy);                                                       \
    if (proxy->Adapter().type() == typeid(CupyAdapter)) {               \
      return (dmlc::get<CupyAdapter>(proxy->Adapter()).Value()).__Fn(); \
    } else if (proxy->Adapter().type() == typeid(CudfAdapter)) {        \
      return (dmlc::get<CudfAdapter>(proxy->Adapter()).Value()).__Fn(); \
    } else {                                                            \
      LOG(FATAL) << "Unknown type: " << proxy->Adapter().type().name(); \
    }                                                                   \
    return 0;                                                           \
  }(__Proxy)

#define DISPATCH_FN(__Proxy, __Fn, ...)                                 \
  [&](DMatrixProxy const* proxy) {                                      \
    if (proxy->Adapter().type() == typeid(CupyAdapter)) {               \
      return __Fn((dmlc::get<CupyAdapter>(proxy->Adapter()).Value()),   \
                  __VA_ARGS__);                                         \
    } else if (proxy->Adapter().type() == typeid(CudfAdapter)) {        \
      return __Fn((dmlc::get<CudfAdapter>(proxy->Adapter()).Value()),   \
                  __VA_ARGS__);                                         \
    } else {                                                            \
      LOG(FATAL) << "Unknown type";                                     \
      return __Fn((dmlc::get<CudfAdapter>(proxy->Adapter()).Value()),   \
                  __VA_ARGS__);                                         \
    }                                                                   \
  }(__Proxy)

void IterativeDeviceDMatrix::Initialize(DataIterHandle iter_handle, float missing, int nthread) {
  // A handle passed to external iterator.
  auto handle = static_cast<std::shared_ptr<DMatrix>*>(proxy_);
  CHECK(handle);
  DMatrixProxy* proxy = static_cast<DMatrixProxy*>(handle->get());
  CHECK(proxy);
  // The external iterator
  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{
    iter_handle, reset_, next_};

  // Accumulate the total number of rows.
  size_t rows = 0;
  size_t row_stride = 0;
  size_t n_batches = 0;

  iter.Reset();
  while (iter.Next()) {
    auto device = proxy->DeviceIdx();
    dh::safe_cuda(cudaSetDevice(device));
    auto batch_rows = DISPATCH_MEM(proxy, NumRows);
    dh::caching_device_vector<size_t> row_counts(batch_rows + 1, 0);
    common::Span<size_t> row_counts_span(row_counts.data().get(),
                                         row_counts.size());
    // FIXME(trivialfis): Use the fact that external iterator knowns data shape to remove
    // this data passing process.
    row_stride = std::max(
        row_stride,
        DISPATCH_FN(proxy, GetRowCounts, row_counts_span, device, missing));
    rows += DISPATCH_MEM(proxy, NumRows);
    n_batches++;
  }
  size_t cols = DISPATCH_MEM(proxy, NumCols);

  // Sketch for all batches.
  iter.Reset();
  common::HistogramCuts cuts;
  common::DenseCuts dense_cuts(&cuts);

  common::SketchContainer sketch_container(batch_param_.max_bin, cols, rows);
  while (iter.Next()) {
    auto device = proxy->DeviceIdx();
    dh::safe_cuda(cudaSetDevice(device));
    // FIXME(trivialfis): Can we do async sketching for all batches here?
    DISPATCH_FN(proxy, common::AdapterDeviceSketch,
                batch_param_.max_bin, missing, rows, cols, device,
                &sketch_container);
  }
  // FIXME(trivialfis): Can we port the pruning and level data structure to GPU?
  dense_cuts.Init(&sketch_container.sketches_, batch_param_.max_bin, rows);

  // Construct the final ellpack page.
  page_.reset(new EllpackPage);
  *(page_->Impl()) = EllpackPageImpl(proxy->DeviceIdx(), cuts, this->IsDense(),
                                     row_stride, rows);
  size_t offset = 0;
  iter.Reset();
  while (iter.Next()) {
    auto device = proxy->DeviceIdx();
    dh::safe_cuda(cudaSetDevice(device));
    auto rows = DISPATCH_MEM(proxy, NumRows);
    dh::caching_device_vector<size_t> row_counts(rows + 1, 0);
    common::Span<size_t> row_counts_span(row_counts.data().get(),
                                         row_counts.size());
    DISPATCH_FN(proxy, GetRowCounts, row_counts_span, device, missing);
    auto new_impl =
        DISPATCH_FN(proxy, EllpackPageImpl,
                    missing, device, this->IsDense(), nthread,
                    DISPATCH_MEM(proxy, IsRowMajor),
                    row_counts_span,
                    row_stride,
                    rows,
                    cols,
                    cuts);
    size_t num_elements = page_->Impl()->Copy(device, &new_impl, offset);
    offset += num_elements;

    proxy->Info().num_row_ = DISPATCH_MEM(proxy, NumRows);
    proxy->Info().num_col_ = cols;
    this->info_.Append(proxy->Info());
  }
}

BatchSet<EllpackPage> IterativeDeviceDMatrix::GetEllpackBatches(const BatchParam& param) {
  CHECK(page_);
  auto begin_iter =
      BatchIterator<EllpackPage>(new SimpleBatchIteratorImpl<EllpackPage>(page_.get()));
  return BatchSet<EllpackPage>(begin_iter);
}
}  // namespace data
}  // namespace xgboost
