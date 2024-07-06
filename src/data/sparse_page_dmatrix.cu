/**
 * Copyright 2021-2024, XGBoost contributors
 */
#include <memory>   // for shared_ptr
#include <utility>  // for move
#include <variant>  // for visit

#include "../common/hist_util.cuh"
#include "../common/hist_util.h"  // for HistogramCuts
#include "batch_utils.h"          // for CheckEmpty, RegenGHist
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

  bst_idx_t row_stride = 0;
  if (!cache_info_.at(id)->written || detail::RegenGHist(batch_param_, param)) {
    this->InitializeSparsePage(ctx);
    // reinitialize the cache
    cache_info_.erase(id);
    MakeCache(this, ".ellpack.page", on_host_, cache_prefix_, &cache_info_);
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

    row_stride = GetRowStride(this);
    this->InitializeSparsePage(ctx);  // reset after use.
    CHECK_NE(row_stride, 0);
    batch_param_ = param;

    auto ft = this->Info().feature_types.ConstDeviceSpan();
    if (on_host_ && std::get_if<EllpackHostPtr>(&ellpack_page_source_) == nullptr) {
      ellpack_page_source_.emplace<EllpackHostPtr>(nullptr);
    }
    std::visit(
        [&](auto&& ptr) {
          ptr.reset();  // make sure resource is released before making new ones.
          using SourceT = typename std::remove_reference_t<decltype(ptr)>::element_type;
          ptr = std::make_shared<SourceT>(this->missing_, ctx->Threads(), this->Info().num_col_,
                                          this->n_batches_, cache_info_.at(id), param,
                                          std::move(cuts), this->IsDense(), row_stride, ft,
                                          this->sparse_page_source_, ctx->Device());
        },
        ellpack_page_source_);
  } else {
    CHECK(sparse_page_source_);
    std::visit([&](auto&& ptr) { ptr->Reset(); }, this->ellpack_page_source_);
  }

  auto batch_set =
      std::visit([this](auto&& ptr) { return BatchSet{BatchIterator<EllpackPage>{ptr}}; },
                 this->ellpack_page_source_);
  return batch_set;
}
}  // namespace xgboost::data
