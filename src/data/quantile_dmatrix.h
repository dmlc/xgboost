/**
 * Copyright 2024, XGBoost Contributors
 */
#pragma once
#include <cstdint>  // for int32_t
#include <memory>   // for shared_ptr
#include <vector>   // for vector

#include "proxy_dmatrix.h"  // for DataIterProxy
#include "xgboost/data.h"   // for DMatrix, BatchIterator, SparsePage
#include "xgboost/span.h"   // for Span

namespace xgboost::common {
class HistogramCuts;
}  // namespace xgboost::common

namespace xgboost::data {
/**
 * @brief Base class for quantile-based DMatrix.
 *
 * `QuantileDMatrix` is an intermediate storage for quantilization results including
 * quantile cuts and histogram index. Quantilization is designed to be performed on stream
 * of data. In practice, we feed batches of data into the QuantileDMatrix.
 *
 * - It's only supported by hist tree method (both CPU and GPU) since approx requires a
 *   re-calculation of quantiles for each iteration. We can fix this by retaining a
 *   reference to the callback if there are feature requests.
 *
 * - The CPU format and the GPU format are different, the former uses a CSR + CSC for
 *   histogram index while the latter uses only Ellpack.
 */
class QuantileDMatrix : public DMatrix {
  template <typename Page>
  static auto InvalidTreeMethod() {
    LOG(FATAL) << "Only `hist` tree method can use `QuantileDMatrix`.";
    return BatchSet<Page>(BatchIterator<Page>(nullptr));
  }

 public:
  DMatrix *Slice(common::Span<std::int32_t const>) final {
    LOG(FATAL) << "Slicing DMatrix is not supported for external memory.";
    return nullptr;
  }
  DMatrix *SliceCol(std::int32_t, std::int32_t) final {
    LOG(FATAL) << "Slicing DMatrix columns is not supported for external memory.";
    return nullptr;
  }

  [[nodiscard]] bool SparsePageExists() const final { return false; }

  BatchSet<SparsePage> GetRowBatches() final {
    LOG(FATAL) << "Not implemented for `QuantileDMatrix`.";
    return BatchSet<SparsePage>(BatchIterator<SparsePage>(nullptr));
  }
  BatchSet<CSCPage> GetColumnBatches(Context const *) final { return InvalidTreeMethod<CSCPage>(); }
  BatchSet<SortedCSCPage> GetSortedColumnBatches(Context const *) final {
    return InvalidTreeMethod<SortedCSCPage>();
  }

  [[nodiscard]] MetaInfo &Info() final { return info_; }
  [[nodiscard]] MetaInfo const &Info() const final { return info_; }

  [[nodiscard]] Context const *Ctx() const final { return &fmat_ctx_; }

 protected:
  Context fmat_ctx_;
  MetaInfo info_;
};

/**
 * @brief Get quantile cuts from reference (Quantile)DMatrix.
 *
 * @param ctx The context of the new DMatrix.
 * @param ref The reference DMatrix.
 * @param n_features Number of features, used for validation only.
 * @param p Batch parameter for the new DMatrix.
 * @param p_cuts Output quantile cuts.
 */
void GetCutsFromRef(Context const *ctx, std::shared_ptr<DMatrix> ref, bst_feature_t n_features,
                    BatchParam p, common::HistogramCuts *p_cuts);

/**
 * @brief Get quantile cuts from ellpack page.
 */
void GetCutsFromEllpack(EllpackPage const &page, common::HistogramCuts *cuts);

namespace cpu_impl {
void SyncFeatureType(Context const *ctx, std::vector<FeatureType> *p_h_ft);

/**
 * @brief Fetch the external data shape.
 */
void GetDataShape(Context const *ctx, DMatrixProxy *proxy,
                  DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext> iter, float missing,
                  ExternalDataInfo *p_info);

/**
 * @brief Create quantile sketch for CPU from an external iterator or from a reference
 *        DMatrix.
 */
void MakeSketches(Context const *ctx,
                  DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext> *iter,
                  DMatrixProxy *proxy, std::shared_ptr<DMatrix> ref, float missing,
                  common::HistogramCuts *cuts, BatchParam const &p, MetaInfo const &info,
                  ExternalDataInfo const &ext_info, std::vector<FeatureType> *p_h_ft);
}  // namespace cpu_impl

namespace cuda_impl {
void MakeSketches(Context const *ctx,
                  DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext> *iter,
                  DMatrixProxy *proxy, std::shared_ptr<DMatrix> ref, BatchParam const &p,
                  float missing, std::shared_ptr<common::HistogramCuts> cuts, MetaInfo const &info,
                  std::int64_t max_quantile_blocks, ExternalDataInfo *p_ext_info);
}  // namespace cuda_impl
}  // namespace xgboost::data
