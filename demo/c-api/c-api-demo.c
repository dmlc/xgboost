/*!
 * Copyright 2019 XGBoost contributors
 *
 * \file c-api-demo.c
 * \brief A simple example of using xgboost C API.
 */

#include <assert.h>
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
  int use_gpu = 0;  // set to 1 to use the GPU for training

  // load the data
  DMatrixHandle dtrain, dtest;
  safe_xgboost(XGDMatrixCreateFromFile("../data/agaricus.txt.train", silent, &dtrain));
  safe_xgboost(XGDMatrixCreateFromFile("../data/agaricus.txt.test", silent, &dtest));

  // create the booster
  BoosterHandle booster;
  DMatrixHandle eval_dmats[2] = {dtrain, dtest};
  safe_xgboost(XGBoosterCreate(eval_dmats, 2, &booster));

  // configure the training
  // available parameters are described here:
  //   https://xgboost.readthedocs.io/en/latest/parameter.html
  safe_xgboost(XGBoosterSetParam(booster, "tree_method", use_gpu ? "gpu_hist" : "hist"));
  if (use_gpu) {
    // set the GPU to use;
    // this is not necessary, but provided here as an illustration
    safe_xgboost(XGBoosterSetParam(booster, "gpu_id", "0"));
  } else {
    // avoid evaluating objective and metric on a GPU
    safe_xgboost(XGBoosterSetParam(booster, "gpu_id", "-1"));
  }

  safe_xgboost(XGBoosterSetParam(booster, "objective", "binary:logistic"));
  safe_xgboost(XGBoosterSetParam(booster, "min_child_weight", "1"));
  safe_xgboost(XGBoosterSetParam(booster, "gamma", "0.1"));
  safe_xgboost(XGBoosterSetParam(booster, "max_depth", "3"));
  safe_xgboost(XGBoosterSetParam(booster, "verbosity", silent ? "0" : "1"));

  // train and evaluate for 10 iterations
  int n_trees = 10;
  const char* eval_names[2] = {"train", "test"};
  const char* eval_result = NULL;
  for (int i = 0; i < n_trees; ++i) {
    safe_xgboost(XGBoosterUpdateOneIter(booster, i, dtrain));
    safe_xgboost(XGBoosterEvalOneIter(booster, i, eval_dmats, eval_names, 2, &eval_result));
    printf("%s\n", eval_result);
  }

  bst_ulong num_feature = 0;
  safe_xgboost(XGBoosterGetNumFeature(booster, &num_feature));
  printf("num_feature: %lu\n", (unsigned long)(num_feature));

  // predict
  bst_ulong out_len = 0;
  const float* out_result = NULL;
  int n_print = 10;

  safe_xgboost(XGBoosterPredict(booster, dtest, 0, 0, 0, &out_len, &out_result));
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

  {
    printf("Dense Matrix Example (XGDMatrixCreateFromMat): ");

    const float values[] = {0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
      1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0};

    DMatrixHandle dmat;
    safe_xgboost(XGDMatrixCreateFromMat(values, 1, 127, 0.0, &dmat));

    bst_ulong out_len = 0;
    const float* out_result = NULL;

    safe_xgboost(XGBoosterPredict(booster, dmat, 0, 0, 0, &out_len,
          &out_result));
    assert(out_len == 1);

    printf("%1.4f \n", out_result[0]);
    safe_xgboost(XGDMatrixFree(dmat));
  }

  {
    printf("Sparse Matrix Example (XGDMatrixCreateFromCSREx): ");

    const size_t indptr[] = {0, 22};
    const unsigned indices[] = {1, 9, 19, 21, 24, 34, 36, 39, 42, 53, 56, 65,
      69, 77, 86, 88, 92, 95, 102, 106, 117, 122};
    const float data[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    DMatrixHandle dmat;
    safe_xgboost(XGDMatrixCreateFromCSREx(indptr, indices, data, 2, 22, 127,
      &dmat));

    bst_ulong out_len = 0;
    const float* out_result = NULL;

    safe_xgboost(XGBoosterPredict(booster, dmat, 0, 0, 0, &out_len,
          &out_result));
    assert(out_len == 1);

    printf("%1.4f \n", out_result[0]);
    safe_xgboost(XGDMatrixFree(dmat));
  }

  {
    printf("Sparse Matrix Example (XGDMatrixCreateFromCSCEx): ");

    const size_t col_ptr[] = {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 8,
      8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 11, 11, 11, 11, 11, 11,
      11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14,
      14, 14, 14, 14, 14, 14, 15, 15, 16, 16, 16, 16, 17, 17, 17, 18, 18, 18,
      18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
      20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22};

    const unsigned indices[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0};

    const float data[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    DMatrixHandle dmat;
    safe_xgboost(XGDMatrixCreateFromCSCEx(col_ptr, indices, data, 128, 22, 1,
      &dmat));

    bst_ulong out_len = 0;
    const float* out_result = NULL;

    safe_xgboost(XGBoosterPredict(booster, dmat, 0, 0, 0, &out_len,
          &out_result));
    assert(out_len == 1);

    printf("%1.4f \n", out_result[0]);
    safe_xgboost(XGDMatrixFree(dmat));
  }

  // free everything
  safe_xgboost(XGBoosterFree(booster));
  safe_xgboost(XGDMatrixFree(dtrain));
  safe_xgboost(XGDMatrixFree(dtest));
  return 0;
}
