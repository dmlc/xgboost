/**
 * Copyright 2021-2024, XGBoost contributors
 */
#include <memory>   // for shared_ptr
#include <utility>  // for move
#include <variant>  // for visit
#include <vector>   // for vector

#include "../common/hist_util.cuh"
#include "../common/hist_util.h"  // for HistogramCuts
#include "batch_utils.h"          // for CheckEmpty, RegenGHist, CachePageRatio
#include "ellpack_page.cuh"
#include "sparse_page_dmatrix.h"
#include "xgboost/context.h"  // for Context
#include "xgboost/data.h"     // for BatchParam

namespace xgboost::data {
BatchSet<EllpackPage> SparsePageDMatrix::GetEllpackBatches(Context const* ctx,
                                                           const BatchParam& param) {
  CHECK(ctx->IsCUDA());
  if (param.Initialized()) {
    CHECK_GE(param.max_bin, 2);
  }
  detail::CheckEmpty(batch_param_, param);
  auto id = MakeCache(this, ".ellpack.page", on_host_, cache_prefix_, &cache_info_);

  if (!cache_info_.at(id)->written || detail::RegenGHist(batch_param_, param)) {
    this->InitializeSparsePage(ctx);
    // reinitialize the cache
    cache_info_.erase(id);
    id = MakeCache(this, ".ellpack.page", on_host_, cache_prefix_, &cache_info_);
    LOG(INFO) << "Generating new a Ellpack page.";
    std::shared_ptr<common::HistogramCuts> cuts;
    if (!param.hess.empty()) {
      cuts = std::make_shared<common::HistogramCuts>(
          common::DeviceSketchWithHessian(ctx, this, param.max_bin, param.hess));
    } else {
      cuts =
          std::make_shared<common::HistogramCuts>(common::DeviceSketch(ctx, this, param.max_bin));
    }
    this->InitializeSparsePage(ctx);  // reset after use.

    std::vector<bst_idx_t> base_rowids, nnz;
    if (this->ext_info_.row_stride == 0) {
      this->ext_info_.row_stride = GetRowStride(this);
    }

    this->InitializeSparsePage(ctx);  // reset after use.
    batch_param_ = param;

    auto ft = this->Info().feature_types.ConstDeviceSpan();
    if (on_host_ && std::get_if<EllpackHostPtr>(&ellpack_page_source_) == nullptr) {
      ellpack_page_source_.emplace<EllpackHostPtr>(nullptr);
    }

    auto cinfo = EllpackCacheInfo{param, /*prefer_device=*/false, /*max_num_device_pages=*/0,
                                  this->missing_};
    CalcCacheMapping(ctx, this->IsDense(), cuts, min_cache_page_bytes_, this->ext_info_, &cinfo);
    CHECK_EQ(cinfo.cache_mapping.size(), this->ext_info_.n_batches)
        << "Page concatenation is only supported by the `ExtMemQuantileDMatrix`.";
    std::visit(
        [&](auto&& ptr) {
          ptr.reset();  // make sure resource is released before making new ones.
          using SourceT = typename std::remove_reference_t<decltype(ptr)>::element_type;
          ptr = std::make_shared<SourceT>(ctx, this->Info().num_col_, this->ext_info_.n_batches,
                                          cache_info_.at(id), std::move(cuts), this->IsDense(),
                                          this->ext_info_.row_stride, ft, this->sparse_page_source_,
                                          cinfo);
        },
        ellpack_page_source_);
  } else {
    CHECK(sparse_page_source_);
    std::visit([&](auto&& ptr) { ptr->Reset(param); }, this->ellpack_page_source_);
  }

  auto batch_set =
      std::visit([this](auto&& ptr) { return BatchSet{BatchIterator<EllpackPage>{ptr}}; },
                 this->ellpack_page_source_);
  return batch_set;
}
}  // namespace xgboost::data
