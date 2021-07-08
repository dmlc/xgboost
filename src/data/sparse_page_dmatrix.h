/*!
 * Copyright 2015-2021 by Contributors
 * \file sparse_page_dmatrix.h
 * \brief External-memory version of DMatrix.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_
#define XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_

#include <xgboost/data.h>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include "ellpack_page_source.h"
#include "sparse_page_source.h"

namespace xgboost {
namespace data {
// Used for external memory.
class SparsePageDMatrix : public DMatrix {
  MetaInfo info_;
  BatchParam batch_param_;
  std::map<std::string, std::shared_ptr<Cache>> cache_info_;

  DMatrixHandle proxy_;
  DataIterHandle iter_;
  DataIterResetCallback *reset_;
  XGDMatrixCallbackNext *next_;

  float missing_;
  int nthreads_;
  std::string cache_prefix_;
  size_t n_batches_ {0};
  // sparse page is the source to other page types, we make a special member function.
  void InitializeSparsePage();
  // Non-virtual version that can be used in constructor
  BatchSet<SparsePage> GetRowBatchesImpl();

 public:
  explicit SparsePageDMatrix(DataIterHandle iter, DMatrixHandle proxy,
                             DataIterResetCallback *reset,
                             XGDMatrixCallbackNext *next, float missing,
                             int32_t nthreads, std::string cache_prefix);

  ~SparsePageDMatrix() override {
    // Clear out all resources
    sparse_page_source_.reset();
    ellpack_page_source_.reset();
    column_source_.reset();
    sorted_column_source_.reset();

    for (auto const &kv : cache_info_) {
      CHECK(kv.second);
      auto n = kv.second->ShardName();
      TryDeleteCacheFile(n);
    }
  }

  MetaInfo& Info() override;

  const MetaInfo& Info() const override;

  bool SingleColBlock() const override { return false; }
  DMatrix *Slice(common::Span<int32_t const>) override {
    LOG(FATAL) << "Slicing DMatrix is not supported for external memory.";
    return nullptr;
  }

 private:
  BatchSet<SparsePage> GetRowBatches() override;
  BatchSet<CSCPage> GetColumnBatches() override;
  BatchSet<SortedCSCPage> GetSortedColumnBatches() override;
  BatchSet<EllpackPage> GetEllpackBatches(const BatchParam& param) override;
  BatchSet<GHistIndexMatrix> GetGradientIndex(const BatchParam&) override;

  // source data pointers.
  std::shared_ptr<SparsePageSource> sparse_page_source_;
  std::shared_ptr<EllpackPageSource> ellpack_page_source_;
  std::shared_ptr<CSCPageSource> column_source_;
  std::shared_ptr<SortedCSCPageSource> sorted_column_source_;
  std::shared_ptr<GHistIndexMatrix> ghist_index_source_;

  bool EllpackExists() const override {
    return static_cast<bool>(ellpack_page_source_);
  }
  bool SparsePageExists() const override {
    return static_cast<bool>(sparse_page_source_);
  }
};

inline std::string MakeId(std::string prefix, SparsePageDMatrix *ptr) {
  std::stringstream ss;
  ss << ptr;
  return prefix + "-" + ss.str();
}

inline std::string
MakeCache(SparsePageDMatrix *ptr, std::string format, std::string prefix,
          std::map<std::string, std::shared_ptr<Cache>> *out) {
  auto &cache_info = *out;
  auto name = MakeId(prefix, ptr);
  auto id = name + format;
  auto it = cache_info.find(id);
  if (it == cache_info.cend()) {
    cache_info[id].reset(new Cache{false, name, format});
  }
  return id;
}
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_
