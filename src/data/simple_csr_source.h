/*!
 * Copyright 2015 by Contributors
 * \file simple_csr_source.h
 * \brief The simplest form of data source, can be used to create DMatrix.
 *  This is an in-memory data structure that holds the data in row oriented format.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SIMPLE_CSR_SOURCE_H_
#define XGBOOST_DATA_SIMPLE_CSR_SOURCE_H_

#include <xgboost/base.h>
#include <xgboost/data.h>
#include <vector>
#include <algorithm>


namespace xgboost {
namespace data {
/*!
 * \brief The simplest form of data holder, can be used to create DMatrix.
 *  This is an in-memory data structure that holds the data in row oriented format.
 * \code
 * std::unique_ptr<DataSource> source(new SimpleCSRSource());
 * // add data to source
 * DMatrix* dmat = DMatrix::Create(std::move(source));
 * \encode
 */
class SimpleCSRSource : public DataSource {
 public:
  // public data members
  // MetaInfo info;  // inheritated from DataSource
  /*! \brief row pointer of CSR sparse storage */
  std::vector<size_t> row_ptr_;
  /*! \brief data in the CSR sparse storage */
  std::vector<RowBatch::Entry> row_data_;
  // functions
  /*! \brief default constructor */
  SimpleCSRSource() : row_ptr_(1, 0), at_first_(true) {}
  /*! \brief destructor */
  virtual ~SimpleCSRSource() {}
  /*! \brief clear the data structure */
  void Clear();
  /*!
   * \brief copy content of data from src
   * \param src source data iter.
   */
  void CopyFrom(DMatrix* src);
  /*!
   * \brief copy content of data from parser, also set the additional information.
   * \param src source data iter.
   * \param info The additional information reflected in the parser.
   */
  void CopyFrom(dmlc::Parser<uint32_t>* src);
  /*!
   * \brief Load data from binary stream.
   * \param fi the pointer to load data from.
   */
  void LoadBinary(dmlc::Stream* fi);
  /*!
   * \brief Save data into binary stream
   * \param fo The output stream.
   */
  void SaveBinary(dmlc::Stream* fo) const;
  // implement Next
  bool Next() override;
  // implement BeforeFirst
  void BeforeFirst() override;
  // implement Value
  const RowBatch &Value() const override;
  /*! \brief magic number used to identify SimpleCSRSource */
  static const int kMagic = 0xffffab01;

 private:
  /*! \brief internal variable, used to support iterator interface */
  bool at_first_;
  /*! \brief */
  RowBatch batch_;
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SIMPLE_CSR_SOURCE_H_
