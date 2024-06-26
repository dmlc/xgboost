/**
 * Copyright 2015-2024, XGBoost Contributors
 * \file sparse_page_dmatrix.h
 * \brief External-memory version of DMatrix.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_
#define XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_

#include <cstdint>  // for uint32_t, int32_t
#include <map>      // for map
#include <memory>   // for shared_ptr
#include <sstream>  // for stringstream
#include <string>   // for string
#include <variant>  // for variant, visit

#include "ellpack_page_source.h"         // for EllpackPageSource, EllpackPageHostSource
#include "gradient_index_page_source.h"  // for GradientIndexPageSource
#include "sparse_page_source.h"          // for SparsePageSource, Cache
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for DMatrix, MetaInfo
#include "xgboost/logging.h"
#include "xgboost/span.h"  // for Span

namespace xgboost::data {
/**
 * @brief DMatrix used for external memory.
 *
 * The external memory is created for controlling memory usage by splitting up data into
 * multiple batches.  However that doesn't mean we will actually process exactly 1 batch
 * at a time, which would be terribly slow considering that we have to loop through the
 * whole dataset for every tree split.  So we use async to pre-fetch pages and let the
 * caller to decide how many batches it wants to process by returning data as a shared
 * pointer. The caller can use async function to process the data or just stage those
 * batches based on its use cases. These two optimizations might defeat the purpose of
 * splitting up dataset since if you stage all the batches then the memory usage might be
 * even worse than using a single batch. As a result, we must control how many batches can
 * be in memory at any given time.
 *
 * Right now the write to the cache is a sequential operation and is blocking. Reading
 * from cache on ther other hand, is async but with a hard coded limit of 3 pages as an
 * heuristic.  So by sparse dmatrix itself there can be only 7 pages in main memory (might
 * be of different types) at the same time: 1 page pending for write, 3 pre-fetched sparse
 * pages, 3 pre-fetched dependent pages.
 *
 * Of course if the caller decides to retain some batches to perform parallel processing,
 * then we might load all pages in memory, which is also considered as a bug in caller's
 * code. So if the algo supports external memory, it must be careful that queue for async
 * call must have an upper limit.
 *
 * Another assumption we make is that the data must be immutable so caller should never
 * change the data.  Sparse page source returns const page to make sure of that.  If you
 * want to change the generated page like Ellpack, pass parameter into `GetBatches` to
 * re-generate them instead of trying to modify the pages in-place.
 *
 * The overall chain of responsibility of external memory DMatrix:
 *
 *    User defined iterator (in Python/C/R) -> Proxy DMatrix -> Sparse page Source ->
 *    Other sources (Like Ellpack) -> Sparse Page DMatrix -> Caller
 *
 * A possible optimization is skipping the sparse page source for `hist` based algorithms
 * similar to the Quantile DMatrix.
 */
class SparsePageDMatrix : public DMatrix {
  MetaInfo info_;
  BatchParam batch_param_;
  std::map<std::string, std::shared_ptr<Cache>> cache_info_;

  DMatrixHandle proxy_;
  DataIterHandle iter_;
  DataIterResetCallback *reset_;
  XGDMatrixCallbackNext *next_;

  float missing_;
  Context fmat_ctx_;
  std::string cache_prefix_;
  bool on_host_{false};
  std::uint32_t n_batches_{0};
  // sparse page is the source to other page types, we make a special member function.
  void InitializeSparsePage(Context const *ctx);
  // Non-virtual version that can be used in constructor
  BatchSet<SparsePage> GetRowBatchesImpl(Context const *ctx);

 public:
  explicit SparsePageDMatrix(DataIterHandle iter, DMatrixHandle proxy, DataIterResetCallback *reset,
                             XGDMatrixCallbackNext *next, float missing, int32_t nthreads,
                             std::string cache_prefix, bool on_host = false);

  ~SparsePageDMatrix() override;

  [[nodiscard]] MetaInfo &Info() override;
  [[nodiscard]] const MetaInfo &Info() const override;
  [[nodiscard]] Context const *Ctx() const override { return &fmat_ctx_; }
  // The only DMatrix implementation that returns false.
  [[nodiscard]] bool SingleColBlock() const override { return false; }
  DMatrix *Slice(common::Span<std::int32_t const>) override {
    LOG(FATAL) << "Slicing DMatrix is not supported for external memory.";
    return nullptr;
  }
  DMatrix *SliceCol(int, int) override {
    LOG(FATAL) << "Slicing DMatrix columns is not supported for external memory.";
    return nullptr;
  }

  [[nodiscard]] bool EllpackExists() const override {
    return std::visit([](auto &&ptr) { return static_cast<bool>(ptr); }, ellpack_page_source_);
  }
  [[nodiscard]] bool GHistIndexExists() const override {
    return static_cast<bool>(ghist_index_source_);
  }
  [[nodiscard]] bool SparsePageExists() const override {
    return static_cast<bool>(sparse_page_source_);
  }
  // For testing, getter for the number of fetches for sparse page source.
  [[nodiscard]] auto SparsePageFetchCount() const {
    return this->sparse_page_source_->FetchCount();
  }

 private:
  BatchSet<SparsePage> GetRowBatches() override;
  BatchSet<CSCPage> GetColumnBatches(Context const *ctx) override;
  BatchSet<SortedCSCPage> GetSortedColumnBatches(Context const *ctx) override;
  BatchSet<EllpackPage> GetEllpackBatches(Context const *ctx, const BatchParam &param) override;
  BatchSet<GHistIndexMatrix> GetGradientIndex(Context const *ctx, const BatchParam &) override;
  BatchSet<ExtSparsePage> GetExtBatches(Context const *, BatchParam const &) override {
    LOG(FATAL) << "Can not obtain a single CSR page for external memory DMatrix";
    return BatchSet<ExtSparsePage>(BatchIterator<ExtSparsePage>(nullptr));
  }

 private:
  // source data pointers.
  std::shared_ptr<SparsePageSource> sparse_page_source_;
  using EllpackDiskPtr = std::shared_ptr<EllpackPageSource>;
  using EllpackHostPtr = std::shared_ptr<EllpackPageHostSource>;
  std::variant<EllpackDiskPtr, EllpackHostPtr> ellpack_page_source_;
  std::shared_ptr<CSCPageSource> column_source_;
  std::shared_ptr<SortedCSCPageSource> sorted_column_source_;
  std::shared_ptr<GradientIndexPageSource> ghist_index_source_;
};

[[nodiscard]] inline std::string MakeId(std::string prefix, SparsePageDMatrix *ptr) {
  std::stringstream ss;
  ss << ptr;
  return prefix + "-" + ss.str();
}

/**
 * @brief Make cache if it doesn't exist yet.
 */
inline std::string MakeCache(SparsePageDMatrix *ptr, std::string format, bool on_host,
                             std::string prefix,
                             std::map<std::string, std::shared_ptr<Cache>> *out) {
  auto &cache_info = *out;
  auto name = MakeId(prefix, ptr);
  auto id = name + format;
  auto it = cache_info.find(id);
  if (it == cache_info.cend()) {
    cache_info[id].reset(new Cache{false, name, format, on_host});
    LOG(INFO) << "Make cache:" << cache_info[id]->ShardName();
  }
  return id;
}
}  // namespace xgboost::data
#endif  // XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_
