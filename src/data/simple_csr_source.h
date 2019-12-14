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

#include <algorithm>
#include <string>
#include <vector>
#include <limits>

namespace xgboost {

class Json;

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
class SimpleCSRSource : public DataSource<SparsePage> {
 public:
  // MetaInfo info;  // inheritated from DataSource
  SparsePage page_;
  /*! \brief default constructor */
  SimpleCSRSource() = default;
  /*! \brief destructor */
  ~SimpleCSRSource() override = default;
  /*! \brief clear the data structure */
  void Clear();
  /*!
   * \brief copy content of data from src
   * \param src source data iter.
   */
  void CopyFrom(DMatrix* src);

  /*!
   * \brief copy content of data from foreign **GPU** columnar buffer.
   * \param interfaces_str JSON representation of cuda array interfaces.
   * \param has_missing Whether did users supply their own missing value.
   * \param missing The missing value set by users.
   */
  void CopyFrom(std::string const& cuda_interfaces_str, bool has_missing,
                bst_float missing = std::numeric_limits<float>::quiet_NaN());
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
  const SparsePage &Value() const override;
  /*! \brief magic number used to identify SimpleCSRSource */
  static const int kMagic = 0xffffab01;

 private:
  /*!
   * \brief copy content of data from foreign GPU columnar buffer.
   * \param columns JSON representation of array interfaces.
   * \param missing specifed missing value
   */
  void FromDeviceColumnar(std::vector<Json> const& columns,
                          bool has_missing = false,
                          float missing = std::numeric_limits<float>::quiet_NaN());
  /*! \brief internal variable, used to support iterator interface */
  bool at_first_{true};
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SIMPLE_CSR_SOURCE_H_
