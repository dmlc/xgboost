/**
 * Copyright 2015-2023, XGBoost Contributors
 * \file sparse_page_dmatrix.h
 * \brief External-memory version of DMatrix.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_
#define XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_

#include <xgboost/data.h>
#include <xgboost/logging.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ellpack_page_source.h"
#include "gradient_index_page_source.h"
#include "sparse_page_source.h"

namespace xgboost {
namespace data {
/**
 * \brief DMatrix used for external memory.
 *
 * The external memory is created for controlling memory usage by splitting up data into
 * multiple batches.  However that doesn't mean we will actually process exact 1 batch at
 * a time, which would be terribly slow considering that we have to loop through the
 * whole dataset for every tree split.  So we use async pre-fetch and let caller to decide
 * how many batches it wants to process by returning data as shared pointer.  The caller
 * can use async function to process the data or just stage those batches, making the
 * decision is out of the scope for sparse page dmatrix.  These 2 optimizations might
 * defeat the purpose of splitting up dataset since if you load all the batches then the
 * memory usage is even worse than using a single batch.  Essentially we need to control
 * how many batches can be in memory at the same time.
 *
 * Right now the write to the cache is sequential operation and is blocking, reading from
 * cache is async but with a hard coded limit of 4 pages as an heuristic.  So by sparse
 * dmatrix itself there can be only 9 pages in main memory (might be of different types)
 * at the same time: 1 page pending for write, 4 pre-fetched sparse pages, 4 pre-fetched
 * dependent pages.  If the caller stops iteration at the middle and start again, then the
 * number of pages in memory can hit 16 due to pre-fetching, but this should be a bug in
 * caller's code (XGBoost doesn't discard a large portion of data at the end, there's not
 * sampling algo that samples only the first portion of data).
 *
 * Of course if the caller decides to retain some batches to perform parallel processing,
 * then we might load all pages in memory, which is also considered as a bug in caller's
 * code.  So if the algo supports external memory, it must be careful that queue for async
 * call must have an upper limit.
 *
 * Another assumption we make is that the data must be immutable so caller should never
 * change the data.  Sparse page source returns const page to make sure of that.  If you
 * want to change the generated page like Ellpack, pass parameter into `GetBatches` to
 * re-generate them instead of trying to modify the pages in-place.
 *
 * A possible optimization is dropping the sparse page once dependent pages like ellpack
 * are constructed and cached.
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
  uint32_t n_batches_{0};
  // sparse page is the source to other page types, we make a special member function.
  void InitializeSparsePage(Context const *ctx);
  // Non-virtual version that can be used in constructor
  BatchSet<SparsePage> GetRowBatchesImpl(Context const *ctx);

 public:
  explicit SparsePageDMatrix(DataIterHandle iter, DMatrixHandle proxy, DataIterResetCallback *reset,
                             XGDMatrixCallbackNext *next, float missing, int32_t nthreads,
                             std::string cache_prefix);

  ~SparsePageDMatrix() override {
    // Clear out all resources before deleting the cache file.
    sparse_page_source_.reset();
    ellpack_page_source_.reset();
    column_source_.reset();
    sorted_column_source_.reset();
    ghist_index_source_.reset();

    for (auto const &kv : cache_info_) {
      CHECK(kv.second);
      auto n = kv.second->ShardName();
      TryDeleteCacheFile(n);
    }
  }

  MetaInfo &Info() override;
  const MetaInfo &Info() const override;
  Context const *Ctx() const override { return &fmat_ctx_; }

  bool SingleColBlock() const override { return false; }
  DMatrix *Slice(common::Span<int32_t const>) override {
    LOG(FATAL) << "Slicing DMatrix is not supported for external memory.";
    return nullptr;
  }
  DMatrix *SliceCol(int, int) override {
    LOG(FATAL) << "Slicing DMatrix columns is not supported for external memory.";
    return nullptr;
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

  // source data pointers.
  std::shared_ptr<SparsePageSource> sparse_page_source_;
  std::shared_ptr<EllpackPageSource> ellpack_page_source_;
  std::shared_ptr<CSCPageSource> column_source_;
  std::shared_ptr<SortedCSCPageSource> sorted_column_source_;
  std::shared_ptr<GradientIndexPageSource> ghist_index_source_;

  bool EllpackExists() const override { return static_cast<bool>(ellpack_page_source_); }
  bool GHistIndexExists() const override { return static_cast<bool>(ghist_index_source_); }
  bool SparsePageExists() const override { return static_cast<bool>(sparse_page_source_); }
};

inline std::string MakeId(std::string prefix, SparsePageDMatrix *ptr) {
  std::stringstream ss;
  ss << ptr;
  return prefix + "-" + ss.str();
}

inline std::string MakeCache(SparsePageDMatrix *ptr, std::string format, std::string prefix,
                             std::map<std::string, std::shared_ptr<Cache>> *out) {
  auto &cache_info = *out;
  auto name = MakeId(prefix, ptr);
  auto id = name + format;
  auto it = cache_info.find(id);
  if (it == cache_info.cend()) {
    cache_info[id].reset(new Cache{false, name, format});
    LOG(INFO) << "Make cache:" << cache_info[id]->ShardName() << std::endl;
  }
  return id;
}
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_
