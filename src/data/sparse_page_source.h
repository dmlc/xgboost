/*!
 *  Copyright (c) 2014 by Contributors
 * \file page_csr_source.h
 *  External memory data source, saved with sparse_batch_page binary format.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SPARSE_PAGE_SOURCE_H_
#define XGBOOST_DATA_SPARSE_PAGE_SOURCE_H_

#include <xgboost/base.h>
#include <xgboost/data.h>
#include <dmlc/threadediter.h>
#include <vector>
#include <algorithm>
#include <string>
#include "sparse_page_writer.h"

namespace xgboost {
namespace data {
/*!
 * \brief External memory data source.
 * \code
 * std::unique_ptr<DataSource> source(new SimpleCSRSource(cache_prefix));
 * // add data to source
 * DMatrix* dmat = DMatrix::Create(std::move(source));
 * \encode
 */
class SparsePageSource : public DataSource {
 public:
  /*!
   * \brief Create source from cache files the cache_prefix.
   * \param cache_prefix The prefix of cache we want to solve.
   */
  explicit SparsePageSource(const std::string& cache_prefix) noexcept(false);
  /*! \brief destructor */
  ~SparsePageSource() override;
  // implement Next
  bool Next() override;
  // implement BeforeFirst
  void BeforeFirst() override;
  // implement Value
  const SparsePage& Value() const override;
  /*!
   * \brief Create source by taking data from parser.
   * \param src source parser.
   * \param cache_info The cache_info of cache file location.
   */
  static void Create(dmlc::Parser<uint32_t>* src,
                     const std::string& cache_info);
  /*!
   * \brief Create source cache by copy content from DMatrix.
   * \param cache_info The cache_info of cache file location.
   */
  static void Create(DMatrix* src,
                     const std::string& cache_info);
  /*!
   * \brief Check if the cache file already exists.
   * \param cache_info The cache prefix of files.
   * \return Whether cache file already exists.
   */
  static bool CacheExist(const std::string& cache_info);
  /*! \brief page size 32 MB */
  static const size_t kPageSize = 32UL << 20UL;
  /*! \brief magic number used to identify Page */
  static const int kMagic = 0xffffab02;

 private:
  /*! \brief number of rows */
  size_t base_rowid_;
  /*! \brief page currently on hold. */
  SparsePage *page_;
  /*! \brief internal clock ptr */
  size_t clock_ptr_;
  /*! \brief file pointer to the row blob file. */
  std::vector<std::unique_ptr<dmlc::SeekStream> > files_;
  /*! \brief Sparse page format file. */
  std::vector<std::unique_ptr<SparsePageFormat> > formats_;
  /*! \brief internal prefetcher. */
  std::vector<std::unique_ptr<dmlc::ThreadedIter<SparsePage> > > prefetchers_;
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_PAGE_SOURCE_H_
