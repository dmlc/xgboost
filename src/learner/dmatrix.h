#ifndef XGBOOST_LEARNER_DMATRIX_H_
#define XGBOOST_LEARNER_DMATRIX_H_
/*!
 * \file dmatrix.h
 * \brief meta data and template data structure 
 *        used for regression/classification/ranking
 * \author Tianqi Chen
 */
#include <vector>
#include "../data.h"

namespace xgboost {
namespace learner {
/*!
 * \brief meta information needed in training, including label, weight
 */
struct MetaInfo {
  /*! \brief number of rows in the data */
  size_t num_row;
  /*! \brief number of columns in the data */
  size_t num_col;
  /*! \brief label of each instance */
  std::vector<float> labels;
  /*!
   * \brief the index of begin and end of a group
   * needed when the learning task is ranking
   */
  std::vector<bst_uint> group_ptr;
  /*! \brief weights of each instance, optional */
  std::vector<float> weights;
  /*!
   * \brief specified root index of each instance,
   *  can be used for multi task setting
   */
  std::vector<unsigned> root_index;
  MetaInfo(void) : num_row(0), num_col(0) {}
  /*! \brief clear all the information */
  inline void Clear(void) {
    labels.clear();
    group_ptr.clear();
    weights.clear();
    root_index.clear();
    num_row = num_col = 0;
  }
  /*! \brief get weight of each instances */
  inline float GetWeight(size_t i) const {
    if (weights.size() != 0) {
      return weights[i];
    } else {
      return 1.0f;
    }
  }
  /*! \brief get root index of i-th instance */
  inline float GetRoot(size_t i) const {
    if (root_index.size() != 0) {
      return static_cast<float>(root_index[i]);
    } else {
      return 0;
    }
  }
  inline void SaveBinary(utils::IStream &fo) const {
    fo.Write(&num_row, sizeof(num_row));
    fo.Write(&num_col, sizeof(num_col));
    fo.Write(labels);
    fo.Write(group_ptr);
    fo.Write(weights);
    fo.Write(root_index);
  }
  inline void LoadBinary(utils::IStream &fi) {
    utils::Check(fi.Read(&num_row, sizeof(num_row)), "MetaInfo: invalid format");
    utils::Check(fi.Read(&num_col, sizeof(num_col)), "MetaInfo: invalid format");
    utils::Check(fi.Read(&labels), "MetaInfo: invalid format");
    utils::Check(fi.Read(&group_ptr), "MetaInfo: invalid format");
    utils::Check(fi.Read(&weights), "MetaInfo: invalid format");
    utils::Check(fi.Read(&root_index), "MetaInfo: invalid format");
  }
  // try to load group information from file, if exists
  inline bool TryLoadGroup(const char* fname, bool silent = false) {
    FILE *fi = fopen64(fname, "r");
    if (fi == NULL) return false;
    group_ptr.push_back(0);
    unsigned nline;
    while (fscanf(fi, "%u", &nline) == 1) {
      group_ptr.push_back(group_ptr.back()+nline);
    }
    if (!silent) {
      printf("%lu groups are loaded from %s\n", group_ptr.size()-1, fname);
    }
    fclose(fi);
    return true;
  }
  // try to load weight information from file, if exists
  inline bool TryLoadWeight(const char* fname, bool silent = false) {
    FILE *fi = fopen64(fname, "r");
    if (fi == NULL) return false;
    float wt;
    while (fscanf(fi, "%f", &wt) == 1) {
      weights.push_back(wt);
    }
    if (!silent) {
      printf("loading weight from %s\n", fname);
    }
    fclose(fi);
    return true;
  }
};

/*!
 * \brief data object used for learning,
 * \tparam FMatrix type of feature data source
 */
template<typename FMatrix>
struct DMatrix {
  /*! 
   * \brief magic number associated with this object 
   *    used to check if it is specific instance
   */
  const int magic;
  /*! \brief meta information about the dataset */
  MetaInfo info;
  /*! \brief feature matrix about data content */
  FMatrix fmat;
  /*! 
   * \brief cache pointer to verify if the data structure is cached in some learner
   *  used to verify if DMatrix is cached
   */
  void *cache_learner_ptr_;
  /*! \brief default constructor */
  explicit DMatrix(int magic) : magic(magic), cache_learner_ptr_(NULL) {}
  // virtual destructor
  virtual ~DMatrix(void){}
};

}  // namespace learner
}  // namespace xgboost
#endif  // XGBOOST_LEARNER_DMATRIX_H_
