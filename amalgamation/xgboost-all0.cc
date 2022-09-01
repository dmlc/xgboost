/*!
 * Copyright 2015-2022 by Contributors.
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
#include "../src/metric/auc.cc"
#include "../src/metric/survival_metric.cc"

// objectives
#include "../src/objective/objective.cc"
#include "../src/objective/regression_obj.cc"
#include "../src/objective/multiclass_obj.cc"
#include "../src/objective/rank_obj.cc"
#include "../src/objective/hinge.cc"
#include "../src/objective/aft_obj.cc"
#include "../src/objective/adaptive.cc"

// gbms
#include "../src/gbm/gbm.cc"
#include "../src/gbm/gbtree.cc"
#include "../src/gbm/gbtree_model.cc"
#include "../src/gbm/gblinear.cc"
#include "../src/gbm/gblinear_model.cc"

// data
#include "../src/data/simple_dmatrix.cc"
#include "../src/data/data.cc"
#include "../src/data/sparse_page_raw_format.cc"
#include "../src/data/ellpack_page.cc"
#include "../src/data/gradient_index.cc"
#include "../src/data/gradient_index_page_source.cc"
#include "../src/data/gradient_index_format.cc"
#include "../src/data/sparse_page_dmatrix.cc"
#include "../src/data/proxy_dmatrix.cc"
#include "../src/data/iterative_dmatrix.cc"

// prediction
#include "../src/predictor/predictor.cc"
#include "../src/predictor/cpu_predictor.cc"

// trees
#include "../src/tree/constraints.cc"
#include "../src/tree/param.cc"
#include "../src/tree/tree_model.cc"
#include "../src/tree/tree_updater.cc"
#include "../src/tree/updater_approx.cc"
#include "../src/tree/updater_colmaker.cc"
#include "../src/tree/updater_prune.cc"
#include "../src/tree/updater_quantile_hist.cc"
#include "../src/tree/updater_refresh.cc"
#include "../src/tree/updater_sync.cc"

// linear
#include "../src/linear/linear_updater.cc"
#include "../src/linear/updater_coordinate.cc"
#include "../src/linear/updater_shotgun.cc"

// global
#include "../src/learner.cc"
#include "../src/logging.cc"
#include "../src/global_config.cc"

// collective
#include "../src/collective/communicator.cc"

// common
#include "../src/common/common.cc"
#include "../src/common/column_matrix.cc"
#include "../src/common/random.cc"
#include "../src/common/charconv.cc"
#include "../src/common/timer.cc"
#include "../src/common/quantile.cc"
#include "../src/common/host_device_vector.cc"
#include "../src/common/hist_util.cc"
#include "../src/common/io.cc"
#include "../src/common/json.cc"
#include "../src/common/pseudo_huber.cc"
#include "../src/common/survival_util.cc"
#include "../src/common/threading_utils.cc"
#include "../src/common/version.cc"

// c_api
#include "../src/c_api/c_api.cc"
#include "../src/c_api/c_api_error.cc"
