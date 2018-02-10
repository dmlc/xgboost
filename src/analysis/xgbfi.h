/*!
* Copyright 2017 by Contributors
* \file xgbfi.h
* \brief xgb feature interactions (xgbfi)
* \author Mathias Müller (Far0n)
*/
#ifndef XGBOOST_ANALYSIS_XGBFI_H_
#define XGBOOST_ANALYSIS_XGBFI_H_

#include <xgboost/learner.h>
#include <string>
#include <vector>


namespace xgbfi {
/*!
* \brief XGBoost Feature Interactions & Importance (Xgbfi)
* \param learner reference to instance of xgboost::Learner
* \param max_fi_depth upper bound for depth of interactions
* \param max_tree_depth upper bound for tree depth to be traversed
* \param max_deepening upper bound for tree deepening
* \param ntrees amount of trees to be traversed
* \param fmap path to fmap file, feature names "F1|F2|.." or empty string
* \return vector of strings formated like "F1|F2|..;stat1;stat2;.."
*/
std::vector<std::string> GetFeatureInteractions(const xgboost::Learner& learner,
                                                int max_fi_depth,
                                                int max_tree_depth,
                                                int max_deepening,
                                                int ntrees,
                                                const char* fmap);
}  // namespace xgbfi
#endif  // XGBOOST_ANALYSIS_XGBFI_H_
