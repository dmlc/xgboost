/*!
 * Copyright 2014 by Contributors
 * \file feature_map.h
 * \brief Feature map data structure to help visualization and model dump.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_FEATURE_MAP_H_
#define XGBOOST_FEATURE_MAP_H_

#include <vector>
#include <string>
#include <cstring>
#include <iostream>

namespace xgboost {
/*!
 * \brief Feature map data structure to help text model dump.
 * TODO(tqchen) consider make it even more lightweight.
 */
class FeatureMap {
 public:
  /*! \brief type of feature maps */
  enum Type {
    kIndicator = 0,
    kQuantitive = 1,
    kInteger = 2,
    kFloat = 3
  };
  /*!
   * \brief load feature map from input stream
   * \param is Input text stream
   */
  inline void LoadText(std::istream& is) { // NOLINT(*)
    int fid;
    std::string fname, ftype;
    while (is >> fid >> fname >> ftype) {
      this->PushBack(fid, fname.c_str(), ftype.c_str());
    }
  }
  /*!
   * \brief push back feature map.
   * \param fid The feature index.
   * \param fname The feature name.
   * \param ftype The feature type.
   */
  inline void PushBack(int fid, const char *fname, const char *ftype) {
    CHECK_EQ(fid, static_cast<int>(names_.size()));
    names_.emplace_back(fname);
    types_.push_back(GetType(ftype));
  }
  /*! \brief clear the feature map */
  inline void Clear() {
    names_.clear();
    types_.clear();
  }
  /*! \return number of known features */
  inline size_t Size() const {
    return names_.size();
  }
  /*! \return name of specific feature */
  inline const char* Name(size_t idx) const {
    CHECK_LT(idx,  names_.size()) << "FeatureMap feature index exceed bound";
    return names_[idx].c_str();
  }
  /*! \return type of specific feature */
  Type type(size_t idx) const {
    CHECK_LT(idx, names_.size()) << "FeatureMap feature index exceed bound";
    return types_[idx];
  }

 private:
  /*!
   * \return feature type enum given name.
   * \param tname The type name.
   * \return The translated type.
   */
  inline static Type GetType(const char* tname) {
    using std::strcmp;
    if (!strcmp("i", tname)) return kIndicator;
    if (!strcmp("q", tname)) return kQuantitive;
    if (!strcmp("int", tname)) return kInteger;
    if (!strcmp("float", tname)) return kFloat;
    LOG(FATAL) << "unknown feature type, use i for indicator and q for quantity";
    return kIndicator;
  }
  /*! \brief name of the feature */
  std::vector<std::string> names_;
  /*! \brief type of the feature */
  std::vector<Type> types_;
};
}  // namespace xgboost
#endif  // XGBOOST_FEATURE_MAP_H_
