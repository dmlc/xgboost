/*!
 * Copyright 2015-2022 by XGBoost Contributors
 * \file simple_dmatrix.h
 * \brief In-memory version of DMatrix.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SIMPLE_DMATRIX_H_
#define XGBOOST_DATA_SIMPLE_DMATRIX_H_

#include <xgboost/base.h>
#include <xgboost/data.h>

#include <memory>
#include <string>

#include "gradient_index.h"

namespace xgboost {
namespace data {
// Used for single batch data.
class SimpleDMatrix : public DMatrix {
 public:
  SimpleDMatrix() = default;
  template <typename AdapterT>
  explicit SimpleDMatrix(AdapterT* adapter, float missing, int nthread);

  explicit SimpleDMatrix(dmlc::Stream* in_stream);
  ~SimpleDMatrix() override = default;

  void SaveToLocalFile(const std::string& fname);

  MetaInfo& Info() override;
  const MetaInfo& Info() const override;
  Context const* Ctx() const override { return &ctx_; }

  bool SingleColBlock() const override { return true; }
  DMatrix* Slice(common::Span<int32_t const> ridxs) override;

  /*! \brief magic number used to identify SimpleDMatrix binary files */
  static const int kMagic = 0xffffab01;

 protected:
  BatchSet<SparsePage> GetRowBatches() override;
  BatchSet<CSCPage> GetColumnBatches() override;
  BatchSet<SortedCSCPage> GetSortedColumnBatches() override;
  BatchSet<EllpackPage> GetEllpackBatches(const BatchParam& param) override;
  BatchSet<GHistIndexMatrix> GetGradientIndex(const BatchParam& param) override;

  MetaInfo info_;
  // Primary storage type
  std::shared_ptr<SparsePage> sparse_page_ = std::make_shared<SparsePage>();
  std::shared_ptr<CSCPage> column_page_{nullptr};
  std::shared_ptr<SortedCSCPage> sorted_column_page_{nullptr};
  std::shared_ptr<EllpackPage> ellpack_page_{nullptr};
  std::shared_ptr<GHistIndexMatrix> gradient_index_{nullptr};
  BatchParam batch_param_;

  bool EllpackExists() const override { return static_cast<bool>(ellpack_page_); }
  bool GHistIndexExists() const override { return static_cast<bool>(gradient_index_); }
  bool SparsePageExists() const override { return true; }

 private:
  Context ctx_;
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SIMPLE_DMATRIX_H_
