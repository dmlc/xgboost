/*!
 * Copyright 2020 XGBoost contributors
 */
#include <memory>
#include <type_traits>
#include <algorithm>

#include "../common/hist_util.cuh"
#include "simple_batch_iterator.h"
#include "iterative_device_dmatrix.h"
#include "sparse_page_source.h"
#include "ellpack_page.cuh"
#include "proxy_dmatrix.h"
#include "device_adapter.cuh"

namespace xgboost {
namespace data {

template <typename Fn>
decltype(auto) Dispatch(DMatrixProxy const* proxy, Fn fn) {
  if (proxy->Adapter().type() == typeid(std::shared_ptr<CupyAdapter>)) {
    auto value = dmlc::get<std::shared_ptr<CupyAdapter>>(
        proxy->Adapter())->Value();
    return fn(value);
  } else if (proxy->Adapter().type() == typeid(std::shared_ptr<CudfAdapter>)) {
    auto value = dmlc::get<std::shared_ptr<CudfAdapter>>(
        proxy->Adapter())->Value();
    return fn(value);
  } else {
    LOG(FATAL) << "Unknown type: " << proxy->Adapter().type().name();
    auto value = dmlc::get<std::shared_ptr<CudfAdapter>>(
        proxy->Adapter())->Value();
    return fn(value);
  }
}

void IterativeDeviceDMatrix::Initialize(DataIterHandle iter_handle, float missing, int nthread) {
  // A handle passed to external iterator.
  auto handle = static_cast<std::shared_ptr<DMatrix>*>(proxy_);
  CHECK(handle);
  DMatrixProxy* proxy = static_cast<DMatrixProxy*>(handle->get());
  CHECK(proxy);
  // The external iterator
  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{
    iter_handle, reset_, next_};

  dh::XGBCachingDeviceAllocator<char> alloc;

  auto num_rows = [&]() {
    return Dispatch(proxy, [](auto const &value) { return value.NumRows(); });
  };
  auto num_cols = [&]() {
    return Dispatch(proxy, [](auto const &value) { return value.NumCols(); });
  };

  size_t row_stride = 0;
  size_t nnz = 0;
  // Sketch for all batches.
  iter.Reset();
  common::HistogramCuts cuts;
  common::DenseCuts dense_cuts(&cuts);

  std::vector<common::SketchContainer> sketch_containers;
  size_t batches = 0;
  size_t accumulated_rows = 0;
  bst_feature_t cols = 0;
  while (iter.Next()) {
    auto device = proxy->DeviceIdx();
    dh::safe_cuda(cudaSetDevice(device));
    if (cols == 0) {
      cols = num_cols();
    } else {
      CHECK_EQ(cols, num_cols()) << "Inconsistent number of columns.";
    }
    sketch_containers.emplace_back(batch_param_.max_bin, num_cols(), num_rows());
    auto* p_sketch = &sketch_containers.back();
    if (proxy->Info().weights_.Size() != 0) {
      proxy->Info().weights_.SetDevice(device);
      Dispatch(proxy, [&](auto const &value) {
          common::AdapterDeviceSketchWeighted(value, batch_param_.max_bin,
                                              proxy->Info(),
                                              missing, device, p_sketch);
        });
    } else {
      Dispatch(proxy, [&](auto const &value) {
          common::AdapterDeviceSketch(value, batch_param_.max_bin, missing,
                                      device, p_sketch);
        });
    }

    auto batch_rows = num_rows();
    accumulated_rows += batch_rows;
    dh::caching_device_vector<size_t> row_counts(batch_rows + 1, 0);
    common::Span<size_t> row_counts_span(row_counts.data().get(),
                                         row_counts.size());
    row_stride =
        std::max(row_stride, Dispatch(proxy, [=](auto const& value) {
                   return GetRowCounts(value, row_counts_span, device, missing);
                 }));
    nnz += thrust::reduce(thrust::cuda::par(alloc),
                          row_counts.begin(), row_counts.end());
    batches++;
  }

  // Merging multiple batches for each column
  std::vector<common::WQSketch::SummaryContainer> summary_array(cols);
  size_t intermediate_num_cuts = std::min(
      accumulated_rows, static_cast<size_t>(batch_param_.max_bin *
                                            common::SketchContainer::kFactor));
  size_t nbytes =
      common::WQSketch::SummaryContainer::CalcMemCost(intermediate_num_cuts);
#pragma omp parallel for num_threads(nthread) if (nthread > 0)
  for (omp_ulong c = 0; c < cols; ++c) {
    for (auto& sketch_batch : sketch_containers) {
      common::WQSketch::SummaryContainer summary;
      sketch_batch.sketches_.at(c).GetSummary(&summary);
      sketch_batch.sketches_.at(c).Init(0, 1);
      summary_array.at(c).Reduce(summary, nbytes);
    }
  }
  sketch_containers.clear();

  // Build the final summary.
  std::vector<common::WQSketch> sketches(cols);
#pragma omp parallel for num_threads(nthread) if (nthread > 0)
  for (omp_ulong c = 0; c < cols; ++c) {
    sketches.at(c).Init(
        accumulated_rows,
        1.0 / (common::SketchContainer::kFactor * batch_param_.max_bin));
    sketches.at(c).PushSummary(summary_array.at(c));
  }
  dense_cuts.Init(&sketches, batch_param_.max_bin, accumulated_rows);
  summary_array.clear();

  this->info_.num_col_ = cols;
  this->info_.num_row_ = accumulated_rows;
  this->info_.num_nonzero_ = nnz;

  // Construct the final ellpack page.
  page_.reset(new EllpackPage);
  *(page_->Impl()) = EllpackPageImpl(proxy->DeviceIdx(), cuts, this->IsDense(),
                                     row_stride, accumulated_rows);

  size_t offset = 0;
  iter.Reset();
  while (iter.Next()) {
    auto device = proxy->DeviceIdx();
    dh::safe_cuda(cudaSetDevice(device));
    auto rows = num_rows();
    dh::caching_device_vector<size_t> row_counts(rows + 1, 0);
    common::Span<size_t> row_counts_span(row_counts.data().get(),
                                         row_counts.size());
    Dispatch(proxy, [=](auto const& value) {
        return GetRowCounts(value, row_counts_span, device, missing);
      });
    auto is_dense = this->IsDense();
    auto new_impl = Dispatch(proxy, [&](auto const &value) {
      return EllpackPageImpl(value, missing, device, is_dense, nthread,
                             row_counts_span, row_stride, rows, cols, cuts);
    });
    size_t num_elements = page_->Impl()->Copy(device, &new_impl, offset);
    offset += num_elements;

    proxy->Info().num_row_ = num_rows();
    proxy->Info().num_col_ = cols;
    if (batches != 1) {
      this->info_.Extend(std::move(proxy->Info()), false);
    }
  }

  if (batches == 1) {
    this->info_ = std::move(proxy->Info());
    CHECK_EQ(proxy->Info().labels_.Size(), 0);
  }

  iter.Reset();
  // Synchronise worker columns
  rabit::Allreduce<rabit::op::Max>(&info_.num_col_, 1);
}

BatchSet<EllpackPage> IterativeDeviceDMatrix::GetEllpackBatches(const BatchParam& param) {
  CHECK(page_);
  auto begin_iter =
      BatchIterator<EllpackPage>(new SimpleBatchIteratorImpl<EllpackPage>(page_.get()));
  return BatchSet<EllpackPage>(begin_iter);
}
}  // namespace data
}  // namespace xgboost
