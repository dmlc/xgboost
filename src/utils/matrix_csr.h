#ifndef XGBOOST_UTILS_MATRIX_CSR_H_
#define XGBOOST_UTILS_MATRIX_CSR_H_
/*!
 * \file matrix_csr.h
 * \brief this file defines some easy to use STL based class for in memory sparse CSR matrix
 * \author Tianqi Chen
 */
#include <vector>
#include <algorithm>
#include "./utils.h"

namespace xgboost {
namespace utils {
/*!
 * \brief a class used to help construct CSR format matrix,
 *        can be used to convert row major CSR to column major CSR
 * \tparam IndexType type of index used to store the index position, usually unsigned or size_t
 * \tparam whether enabling the usage of aclist, this option must be enabled manually
 */
template<typename IndexType, bool UseAcList = false, typename SizeType = size_t>
struct SparseCSRMBuilder {
 private:
  /*! \brief dummy variable used in the indicator matrix construction */
  std::vector<size_t> dummy_aclist;
  /*! \brief pointer to each of the row */
  std::vector<SizeType> &rptr;
  /*! \brief index of nonzero entries in each row */
  std::vector<IndexType> &findex;
  /*! \brief a list of active rows, used when many rows are empty */
  std::vector<size_t> &aclist;

 public:
  SparseCSRMBuilder(std::vector<SizeType> &p_rptr,
                    std::vector<IndexType> &p_findex)
      :rptr(p_rptr), findex(p_findex), aclist(dummy_aclist) {
    Assert(!UseAcList, "enabling bug");
  }
  /*! \brief use with caution! rptr must be cleaned before use */
  SparseCSRMBuilder(std::vector<SizeType> &p_rptr,
                    std::vector<IndexType> &p_findex,
                    std::vector<size_t> &p_aclist)
      :rptr(p_rptr), findex(p_findex), aclist(p_aclist) {
    Assert(UseAcList, "must manually enable the option use aclist");
  }

 public:
  /*!
   * \brief step 1: initialize the number of rows in the data, not necessary exact
   * \nrows number of rows in the matrix, can be smaller than expected
   */
  inline void InitBudget(size_t nrows = 0) {
    if (!UseAcList) {
      rptr.clear();
      rptr.resize(nrows + 1, 0);
    } else {
      Assert(nrows + 1 == rptr.size(), "rptr must be initialized already");
      this->Cleanup();
    }
  }
  /*!
   * \brief step 2: add budget to each rows, this function is called when aclist is used
   * \param row_id the id of the row
   * \param nelem  number of element budget add to this row
   */
  inline void AddBudget(size_t row_id, SizeType nelem = 1) {
    if (rptr.size() < row_id + 2) {
      rptr.resize(row_id + 2, 0);
    }
    if (UseAcList) {
      if (rptr[row_id + 1] == 0) aclist.push_back(row_id);
    }
    rptr[row_id + 1] += nelem;
  }
  /*! \brief step 3: initialize the necessary storage */
  inline void InitStorage(void) {
    // initialize rptr to be beginning of each segment
    size_t start = 0;
    if (!UseAcList) {
      for (size_t i = 1; i < rptr.size(); i++) {
        size_t rlen = rptr[i];
        rptr[i] = start;
        start += rlen;
      }
    } else {
      // case with active list
      std::sort(aclist.begin(), aclist.end());
      for (size_t i = 0; i < aclist.size(); i++) {
        size_t ridx = aclist[i];
        size_t rlen = rptr[ridx + 1];
        rptr[ridx + 1] = start;
        // set previous rptr to right position if previous feature is not active
        if (i == 0 || ridx != aclist[i - 1] + 1) rptr[ridx] = start;
        start += rlen;
      }
    }
    findex.resize(start);
  }
  /*!
   * \brief step 4:
   * used in indicator matrix construction, add new
   * element to each row, the number of calls shall be exactly same as add_budget
   */
  inline void PushElem(size_t row_id, IndexType col_id) {
    SizeType &rp = rptr[row_id + 1];
    findex[rp++] = col_id;
  }
  /*!
   * \brief step 5: only needed when aclist is used
   * clean up the rptr for next usage
   */
  inline void Cleanup(void) {
    Assert(UseAcList, "this function can only be called use AcList");
    for (size_t i = 0; i < aclist.size(); i++) {
      const size_t ridx = aclist[i];
      rptr[ridx] = 0; rptr[ridx + 1] = 0;
    }
    aclist.clear();
  }
};

}  // namespace utils
}  // namespace xgboost
#endif
