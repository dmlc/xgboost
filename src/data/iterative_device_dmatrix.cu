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

  std::vector<common::SketchContainer> sketch_containers;
  size_t batches = 0;
  size_t accumulated_rows = 0;
  bst_feature_t cols = 0;
  int32_t device = GenericParameter::kCpuId;
  int32_t current_device;
  dh::safe_cuda(cudaGetDevice(&current_device));
  auto get_device = [&]() -> int32_t {
    int32_t d = (device == GenericParameter::kCpuId) ? current_device : device;
    CHECK_NE(d, GenericParameter::kCpuId);
    return d;
  };

  while (iter.Next()) {
    device = proxy->DeviceIdx();
    CHECK_LT(device, common::AllVisibleGPUs());
    dh::safe_cuda(cudaSetDevice(get_device()));
    if (cols == 0) {
      cols = num_cols();
      rabit::Allreduce<rabit::op::Max>(&cols, 1);
    } else {
      CHECK_EQ(cols, num_cols()) << "Inconsistent number of columns.";
    }
    sketch_containers.emplace_back(proxy->Info().feature_types,
                                   batch_param_.max_bin, cols, num_rows(), get_device());
    auto* p_sketch = &sketch_containers.back();
    proxy->Info().weights_.SetDevice(get_device());
    Dispatch(proxy, [&](auto const &value) {
        common::AdapterDeviceSketch(value, batch_param_.max_bin,
                                    proxy->Info(), missing, p_sketch);
      });
    auto batch_rows = num_rows();
    accumulated_rows += batch_rows;
    dh::caching_device_vector<size_t> row_counts(batch_rows + 1, 0);
    common::Span<size_t> row_counts_span(row_counts.data().get(),
                                         row_counts.size());
    row_stride = std::max(row_stride, Dispatch(proxy, [=](auto const &value) {
          return GetRowCounts(value, row_counts_span,
                              get_device(), missing);
        }));
    nnz += thrust::reduce(thrust::cuda::par(alloc), row_counts.begin(),
                          row_counts.end());
    batches++;
  }
  iter.Reset();
  dh::safe_cuda(cudaSetDevice(get_device()));
  HostDeviceVector<FeatureType> ft;
  common::SketchContainer final_sketch(
      sketch_containers.empty() ? ft : sketch_containers.front().FeatureTypes(),
      batch_param_.max_bin, cols, accumulated_rows, get_device());
  for (auto const& sketch : sketch_containers) {
    final_sketch.Merge(sketch.ColumnsPtr(), sketch.Data());
    final_sketch.FixError();
  }
  sketch_containers.clear();
  sketch_containers.shrink_to_fit();

  common::HistogramCuts cuts;
  final_sketch.MakeCuts(&cuts);

  this->info_.num_col_ = cols;
  this->info_.num_row_ = accumulated_rows;
  this->info_.num_nonzero_ = nnz;

  auto init_page = [this, &proxy, &cuts, row_stride, accumulated_rows,
                    get_device]() {
    if (!page_) {
      // Should be put inside the while loop to protect against empty batch.  In
      // that case device id is invalid.
      page_.reset(new EllpackPage);
      *(page_->Impl()) = EllpackPageImpl(get_device(), cuts, this->IsDense(),
                                         row_stride, accumulated_rows);
    }
  };

  // Construct the final ellpack page.
  size_t offset = 0;
  iter.Reset();
  size_t n_batches_for_verification = 0;
  while (iter.Next()) {
    init_page();
    dh::safe_cuda(cudaSetDevice(get_device()));
    auto rows = num_rows();
    dh::caching_device_vector<size_t> row_counts(rows + 1, 0);
    common::Span<size_t> row_counts_span(row_counts.data().get(),
                                         row_counts.size());
    Dispatch(proxy, [=](auto const& value) {
        return GetRowCounts(value, row_counts_span, get_device(), missing);
      });
    auto is_dense = this->IsDense();
    auto new_impl = Dispatch(proxy, [&](auto const &value) {
        return EllpackPageImpl(value, missing, get_device(), is_dense, nthread,
                               row_counts_span, row_stride, rows, cols, cuts);
    });
    size_t num_elements = page_->Impl()->Copy(get_device(), &new_impl, offset);
    offset += num_elements;

    proxy->Info().num_row_ = num_rows();
    proxy->Info().num_col_ = cols;
    if (batches != 1) {
      this->info_.Extend(std::move(proxy->Info()), false);
    }
    n_batches_for_verification++;
  }
  CHECK_EQ(batches, n_batches_for_verification)
      << "Different number of batches returned between 2 iterations";

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
