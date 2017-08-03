/*!
 * Copyright (c) 2017 by Contributors
 * \file parameter_wrapper.h
 * \brief a thin wrapper for DMLC parameter, in order to keep track of unused
 *        arguments
 * \author Philip Cho
 */
#ifndef XGBOOST_PARAMETER_WRAPPER_H_
#define XGBOOST_PARAMETER_WRAPPER_H_

#include <xgboost/learner.h>

namespace xgboost {

template<typename PType>
struct TrackedParameter : public dmlc::Parameter<PType> {
  /*!
   * \brief initialize the parameter by keyword arguments.
   *  This is same as Init, but allow unknown arguments.
   *
   * \param kwargs map of keyword arguments, or vector of pairs
   * \tparam Container container type
   * \throw ParamError when something go wrong.
   * \return vector of pairs of unknown arguments.
   */
  template<typename Container>
  inline std::vector<std::pair<std::string, std::string> >
  InitAllowUnknown(const Container &kwargs) {
    auto unknown = dmlc::Parameter<PType>::InitAllowUnknown(kwargs);
    auto fields = dmlc::Parameter<PType>::__DICT__();
    std::vector<std::string> unused_str;
    std::vector<std::string> fields_str;
    std::transform(unknown.begin(), unknown.end(),
      std::back_inserter(unused_str),
      [] (const std::pair<std::string, std::string>& pair) {
        return pair.first;
      });
    std::transform(fields.begin(), fields.end(),
      std::back_inserter(fields_str),
      [] (const std::pair<std::string, std::string>& pair) {
        return pair.first;
      });
    Learner::RegisterUnusedArgs(unused_str);
    Learner::RegisterValidArgs(fields_str);
    return unknown;
  }
};

}  // namespace xgboost

#endif  // XGBOOST_PARAMETER_WRAPPER_H_
