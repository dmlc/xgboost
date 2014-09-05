#ifndef XGBOOST_LEARNER_DMATRIX_H_
#define XGBOOST_LEARNER_DMATRIX_H_
/*!
 * \file dmatrix.h
 * \brief meta data and template data structure 
 *        used for regression/classification/ranking
 * \author Tianqi Chen
 */
#include <vector>
#include <cstring>
#include "../data.h"
#include "../utils/io.h"
namespace xgboost {
namespace learner {
/*!
 * \brief meta information needed in training, including label, weight
 */
struct MetaInfo {
  /*! 
   * \brief information needed by booster 
   * BoosterInfo does not implement save and load,
   * all serialization is done in MetaInfo
   */
  BoosterInfo info;
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
   * \brief initialized margins,
   * if specified, xgboost will start from this init margin
   * can be used to specify initial prediction to boost from
   */
  std::vector<float> base_margin;
  /*! \brief version flag, used to check version of this info */
  static const int kVersion = 0;
  // constructor
  MetaInfo(void) {}
  /*! \return number of rows in dataset */
  inline size_t num_row(void) const {
    return info.num_row;
  }
  /*! \return number of columns in dataset */
  inline size_t num_col(void) const {
    return info.num_col;
  }
  /*! \brief clear all the information */
  inline void Clear(void) {
    labels.clear();
    group_ptr.clear();
    weights.clear();
    info.root_index.clear();
    base_margin.clear();
    info.num_row = info.num_col = 0;
  }
  /*! \brief get weight of each instances */
  inline float GetWeight(size_t i) const {
    if (weights.size() != 0) {
      return weights[i];
    } else {
      return 1.0f;
    }
  }
  inline void SaveBinary(utils::IStream &fo) const {
    int version = kVersion;
    fo.Write(&version, sizeof(version));
    fo.Write(&info.num_row, sizeof(info.num_row));
    fo.Write(&info.num_col, sizeof(info.num_col));
    fo.Write(labels);
    fo.Write(group_ptr);
    fo.Write(weights);
    fo.Write(info.root_index);
    fo.Write(base_margin);
  }
  inline void LoadBinary(utils::IStream &fi) {
    int version;
    utils::Check(fi.Read(&version, sizeof(version)) != 0, "MetaInfo: invalid format");
    utils::Check(fi.Read(&info.num_row, sizeof(info.num_row)) != 0, "MetaInfo: invalid format");
    utils::Check(fi.Read(&info.num_col, sizeof(info.num_col)) != 0, "MetaInfo: invalid format");
    utils::Check(fi.Read(&labels), "MetaInfo: invalid format");
    utils::Check(fi.Read(&group_ptr), "MetaInfo: invalid format");
    utils::Check(fi.Read(&weights), "MetaInfo: invalid format");
    utils::Check(fi.Read(&info.root_index), "MetaInfo: invalid format");
    utils::Check(fi.Read(&base_margin), "MetaInfo: invalid format");
  }
  // try to load group information from file, if exists
  inline bool TryLoadGroup(const char* fname, bool silent = false) {
    using namespace std;
    FILE *fi = fopen64(fname, "r");
    if (fi == NULL) return false;
    group_ptr.push_back(0);
    unsigned nline;
    while (fscanf(fi, "%u", &nline) == 1) {
      group_ptr.push_back(group_ptr.back()+nline);
    }
    if (!silent) {
      utils::Printf("%u groups are loaded from %s\n",
                    static_cast<unsigned>(group_ptr.size()-1), fname);
    }
    fclose(fi);
    return true;
  }
  inline std::vector<float>& GetFloatInfo(const char *field) {
    using namespace std;
    if (!strcmp(field, "label")) return labels;
    if (!strcmp(field, "weight")) return weights;
    if (!strcmp(field, "base_margin")) return base_margin;
    utils::Error("unknown field %s", field);
    return labels;
  }
  inline const std::vector<float>& GetFloatInfo(const char *field) const {
    return ((MetaInfo*)this)->GetFloatInfo(field);
  }
  inline std::vector<unsigned> &GetUIntInfo(const char *field) {
    using namespace std;
    if (!strcmp(field, "root_index")) return info.root_index;
    if (!strcmp(field, "fold_index")) return info.fold_index;
    utils::Error("unknown field %s", field);
    return info.root_index;
  }
  inline const std::vector<unsigned> &GetUIntInfo(const char *field) const {
    return ((MetaInfo*)this)->GetUIntInfo(field);
  }
  // try to load weight information from file, if exists
  inline bool TryLoadFloatInfo(const char *field, const char* fname, bool silent = false) {
    using namespace std;
    std::vector<float> &data = this->GetFloatInfo(field);
    FILE *fi = fopen64(fname, "r");
    if (fi == NULL) return false;
    float wt;
    while (fscanf(fi, "%f", &wt) == 1) {
      data.push_back(wt);
    }
    if (!silent) {
      utils::Printf("loading %s from %s\n", field, fname);
    }
    fclose(fi);
    return true;
  }
};

/*!
 * \brief data object used for learning,
 * \tparam FMatrix type of feature data source
 */
struct DMatrix {
  /*! 
   * \brief magic number associated with this object 
   *    used to check if it is specific instance
   */
  const int magic;
  /*! \brief meta information about the dataset */
  MetaInfo info;
  /*! 
   * \brief cache pointer to verify if the data structure is cached in some learner
   *  used to verify if DMatrix is cached
   */
  void *cache_learner_ptr_;
  /*! \brief default constructor */
  explicit DMatrix(int magic) : magic(magic), cache_learner_ptr_(NULL) {}
  /*! \brief get feature matrix about data content */
  virtual IFMatrix *fmat(void) const = 0;
  // virtual destructor
  virtual ~DMatrix(void){}
};

}  // namespace learner
}  // namespace xgboost
#endif  // XGBOOST_LEARNER_DMATRIX_H_
