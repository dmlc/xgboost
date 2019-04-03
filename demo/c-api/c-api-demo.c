/** 
 * @file A demo for xgboost C API.
 */

#include <stdio.h>
#include <stdlib.h>
#include <xgboost/c_api.h>

#define safe_xgboost(call) {                                            \
int err = (call);                                                       \
if (err != 0) {                                                         \
  fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
  exit(1);                                                              \
}                                                                       \
}

int main(int argc, char** argv) {
  int silent = 0;
  
  // load the data
  DMatrixHandle dtrain, dtest;
  safe_xgboost(XGDMatrixCreateFromFile("../data/agaricus.txt.train", silent, &dtrain));
  safe_xgboost(XGDMatrixCreateFromFile("../data/agaricus.txt.test", silent, &dtest));
  
  // create the booster
  BoosterHandle booster;
  DMatrixHandle eval_dmats[2] = {dtrain, dtest};
  const char* eval_names[2] = {"train", "test"};
  const char* test_name = "test";
  safe_xgboost(XGBoosterCreate(&dtrain, 1, &booster));

  // configure the training
  safe_xgboost(XGBoosterSetParam(booster, "tree_method", "hist"));
  safe_xgboost(XGBoosterSetParam(booster, "objective", "binary:logistic"));
  safe_xgboost(XGBoosterSetParam(booster, "min_child_weight", "1"));
  safe_xgboost(XGBoosterSetParam(booster, "gamma", "0.1"));
  safe_xgboost(XGBoosterSetParam(booster, "max_depth", "3"));
  safe_xgboost(XGBoosterSetParam(booster, "verbosity", silent ? "0" : "1"));
  
  // train and evaluate for 10 iterations
  int n_trees = 10;
  const char* eval_result = NULL;
  for (int i = 0; i < n_trees; ++i) {
    safe_xgboost(XGBoosterUpdateOneIter(booster, i, dtrain));
    safe_xgboost(XGBoosterEvalOneIter(booster, i, eval_dmats, eval_names, 2, &eval_result));
    printf("%s\n", eval_result);
  }

  // predict
  bst_ulong out_len = 0;
  const float* out_result = NULL;
  int n_print = 10;

  safe_xgboost(XGBoosterPredict(booster, dtest, 0, 0, &out_len, &out_result));
  printf("y_pred: ");
  for (int i = 0; i < n_print; ++i) {
    printf("%1.4f ", out_result[i]);
  }
  printf("\n");

  // print true labels
  safe_xgboost(XGDMatrixGetFloatInfo(dtest, "label", &out_len, &out_result));
  printf("y_test: ");
  for (int i = 0; i < n_print; ++i) {
    printf("%1.4f ", out_result[i]);
  }
  printf("\n");

  // free everything
  safe_xgboost(XGBoosterFree(booster));
  safe_xgboost(XGDMatrixFree(dtrain));
  safe_xgboost(XGDMatrixFree(dtest));
  return 0;
}
