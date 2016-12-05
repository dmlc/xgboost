/*!
 * Copyright 2015 by Contributors.
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

// objectives
#include "../src/objective/objective.cc"
#include "../src/objective/regression_obj.cc"
#include "../src/objective/multiclass_obj.cc"
#include "../src/objective/rank_obj.cc"

// gbms
#include "../src/gbm/gbm.cc"
#include "../src/gbm/gbtree.cc"
#include "../src/gbm/gblinear.cc"

// data
#include "../src/data/data.cc"
#include "../src/data/simple_csr_source.cc"
#include "../src/data/simple_dmatrix.cc"
#include "../src/data/sparse_page_raw_format.cc"

#if DMLC_ENABLE_STD_THREAD
#include "../src/data/sparse_page_source.cc"
#include "../src/data/sparse_page_dmatrix.cc"
#include "../src/data/sparse_page_writer.cc"
#endif

// tress
#include "../src/tree/tree_model.cc"
#include "../src/tree/tree_updater.cc"
#include "../src/tree/updater_colmaker.cc"
#include "../src/tree/updater_prune.cc"
#include "../src/tree/updater_refresh.cc"
#include "../src/tree/updater_sync.cc"
#include "../src/tree/updater_histmaker.cc"
#include "../src/tree/updater_skmaker.cc"

// global
#include "../src/learner.cc"
#include "../src/logging.cc"
#include "../src/common/common.cc"

// c_api
#include "../src/c_api/c_api.cc"
#include "../src/c_api/c_api_error.cc"
