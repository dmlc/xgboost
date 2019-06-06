/*!
 * Copyright 2019 XGBoost contributors
 *
 * \file dmatrix-api-demo.c
 * \brief An example of using xgboost C API to create a DMatrix using callbacks.
 */

#include <stdio.h>
#include <stdlib.h>
#include <xgboost/c_api.h>

#define safe_xgboost(call)                                                   \
  {                                                                          \
    int err = (call);                                                        \
    if (err != 0) {                                                          \
      fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, \
              XGBGetLastError());                                            \
      exit(1);                                                               \
    }                                                                        \
  }

#define kNumElements 4
#define kNumRows 2
#define kNumColumns 2

struct MyDataContainer {
  float* data;
  int* index;
  size_t* offset;
};

int MyDMatrixCallback(DataIterHandle data_handle, size_t* offset,
                      DataElement* elements) {
  struct MyDataContainer* data = (struct MyDataContainer*)data_handle;
  for (int i = 0; i < kNumRows + 1; i++) {
    offset[i] = data->offset[i];
  }

  for (int i = 0; i < kNumElements; i++) {
    DataElement e;
    e.index = data->index[i];
    e.fvalue = data->data[i];
    elements[i] = e;
  }
  return 0;
}

int main(int argc, char** argv) {
  float data[kNumElements] = {0.1f, 0.2f, 0.3f, 0.4f};
  int index[kNumElements] = {0, 1, 0, 1};
  int64_t offset[kNumRows + 1] = {0, 2, 4};
  struct MyDataContainer container;
  container.data = data;
  container.index = index;
  container.offset = offset;
  DMatrixHandle dtrain;
  safe_xgboost(XGDMatrixCreateFromCallBackDirect(&container, MyDMatrixCallback,
                                                 &dtrain, kNumRows,
                                                 kNumElements, kNumColumns));
  long num_rows;
  long num_columns;
  safe_xgboost(XGDMatrixNumRow(dtrain, &num_rows));
  safe_xgboost(XGDMatrixNumCol(dtrain, &num_columns));
  printf("DMatrix created with %lu rows, %lu columns.\n", num_rows,
         num_columns);
  return 0;
}
