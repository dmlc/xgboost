#ifndef XGBOOST_LEARNER_DMATRIX_H_
#define XGBOOST_LEARNER_DMATRIX_H_
/*!
 * \file dmatrix.h
 * \brief meta data and template data structure 
 *        used for regression/classification/ranking
 * \author Tianqi Chen
 */
#include "../data.h"

namespace xgboost {
namespace learner {
/*! 
 * \brief meta information needed in training, including label, weight
 */
struct MetaInfo {
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
  /*! \brief get weight of each instances */
  inline float GetWeight(size_t i) const {
    if(weights.size() != 0) {
      return weights[i];
    } else {
      return 1.0f;
    }
  }
  /*! \brief get root index of i-th instance */
  inline float GetRoot(size_t i) const {
    if(root_index.size() != 0) {
      return static_cast<float>(root_index[i]);
    } else {
      return 0;
    }
  }
  inline void SaveBinary(utils::IStream &fo) {
    fo.Write(labels);
    fo.Write(group_ptr);
    fo.Write(weights);
    fo.Write(root_index);
  }
  inline void LoadBinary(utils::IStream &fi) {
    utils::Check(fi.Read(&labels), "MetaInfo: invalid format");
    utils::Check(fi.Read(&group_ptr), "MetaInfo: invalid format");
    utils::Check(fi.Read(&weights), "MetaInfo: invalid format");
    utils::Check(fi.Read(&root_index), "MetaInfo: invalid format");
  }
};

/*! 
 * \brief data object used for learning,
 * \tparam FMatrix type of feature data source
 */
template<typename FMatrix>
struct DMatrix {
  /*! \brief meta information about the dataset */
  MetaInfo info;
  /*! \brief number of rows in the DMatrix */
  size_t num_row;
  /*! \brief feature matrix about data content */
  FMatrix fmat;
  /*! 
   * \brief cache pointer to verify if the data structure is cached in some learner
   *  used to verify if DMatrix is cached
   */
  void *cache_learner_ptr_;
  /*! \brief default constructor */
  DMatrix(void) : cache_learner_ptr_(NULL) {}
};

}  // namespace learner
}  // namespace xgboost
#endif  // XGBOOST_LEARNER_DMATRIX_H_
