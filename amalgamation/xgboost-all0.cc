/*!
 * Copyright 2015-2019 by Contributors.
 * \brief XGBoost Amalgamation.
 *  This offers an alternative way to compile the entire library from this single file.
 *
 *  Example usage command.
 *  - $(CXX) -std=c++0x -fopenmp -o -shared libxgboost.so xgboost-all0.cc -ldmlc -lrabit
 *
 * \author Tianqi Chen.
 */

// metrics
#include "../src/metric/metric.cc"
#include "../src/metric/elementwise_metric.cc"
#include "../src/metric/multiclass_metric.cc"
#include "../src/metric/rank_metric.cc"
#include "../src/metric/survival_metric.cc"

// objectives
#include "../src/objective/objective.cc"
#include "../src/objective/regression_obj.cc"
#include "../src/objective/multiclass_obj.cc"
#include "../src/objective/rank_obj.cc"
#include "../src/objective/hinge.cc"
#include "../src/objective/aft_obj.cc"

// gbms
#include "../src/gbm/gbm.cc"
#include "../src/gbm/gbtree.cc"
#include "../src/gbm/gbtree_model.cc"
#include "../src/gbm/gblinear.cc"
#include "../src/gbm/gblinear_model.cc"

// data
#include "../src/data/data.cc"
#include "../src/data/simple_dmatrix.cc"
#include "../src/data/sparse_page_raw_format.cc"
#include "../src/data/ellpack_page.cc"
#include "../src/data/ellpack_page_source.cc"

// prediction
#include "../src/predictor/predictor.cc"
#include "../src/predictor/cpu_predictor.cc"

#if DMLC_ENABLE_STD_THREAD
#include "../src/data/sparse_page_dmatrix.cc"
#endif

// trees
#include "../src/tree/param.cc"
#include "../src/tree/split_evaluator.cc"
#include "../src/tree/tree_model.cc"
#include "../src/tree/tree_updater.cc"
#include "../src/tree/updater_colmaker.cc"
#include "../src/tree/updater_quantile_hist.cc"
#include "../src/tree/updater_prune.cc"
#include "../src/tree/updater_refresh.cc"
#include "../src/tree/updater_sync.cc"
#include "../src/tree/updater_histmaker.cc"
#include "../src/tree/updater_skmaker.cc"
#include "../src/tree/constraints.cc"

// linear
#include "../src/linear/linear_updater.cc"
#include "../src/linear/updater_coordinate.cc"
#include "../src/linear/updater_shotgun.cc"

// global
#include "../src/learner.cc"
#include "../src/logging.cc"
#include "../src/common/common.cc"
#include "../src/common/timer.cc"
#include "../src/common/host_device_vector.cc"
#include "../src/common/hist_util.cc"
#include "../src/common/json.cc"
#include "../src/common/io.cc"
#include "../src/common/survival_util.cc"
#include "../src/common/probability_distribution.cc"
#include "../src/common/version.cc"

// c_api
#include "../src/c_api/c_api.cc"
#include "../src/c_api/c_api_error.cc"
