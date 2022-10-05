/*!
 * Copyright 2021 XGBoost contributors
 *
 * \brief A simple example of using xgboost data callback API.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <xgboost/c_api.h>

#define safe_xgboost(err)                                                      \
  if ((err) != 0) {                                                            \
    fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #err,      \
            XGBGetLastError());                                                \
    exit(1);                                                                   \
  }

#define N_BATCHS 32
#define BATCH_LEN 512

/* Shorthands. */
typedef DMatrixHandle DMatrix;
typedef BoosterHandle Booster;

typedef struct _DataIter {
  /* Data of each batch. */
  float **data;
  /* Labels of each batch */
  float **labels;
  /* Length of each batch. */
  size_t *lengths;
  /* Total number of batches. */
  size_t n;
  /* Current iteration. */
  size_t cur_it;

  /* Private fields */
  DMatrix _proxy;
  char _array[128];
} DataIter;

#define safe_malloc(ptr)                                                       \
  if ((ptr) == NULL) {                                                         \
    fprintf(stderr, "%s:%d: Failed to allocate memory.\n", __FILE__,           \
            __LINE__);                                                         \
    exit(1);                                                                   \
  }

/**
 * Initialize with random data for demo. In practice the data should be loaded
 * from external memory.  We just demonstrate how to use the iterator in
 * XGBoost.
 *
 * \param batch_size  Number of elements for each batch.  The demo here is only using 1
 *                    column.
 * \param n_batches   Number of batches.
 */
void DataIterator_Init(DataIter *self, size_t batch_size, size_t n_batches) {
  self->n = n_batches;

  self->lengths = (size_t *)malloc(self->n * sizeof(size_t));
  safe_malloc(self->lengths);
  for (size_t i = 0; i < self->n; ++i) {
    self->lengths[i] = batch_size;
  }

  self->data = (float **)malloc(self->n * sizeof(float *));
  safe_malloc(self->data);
  self->labels = (float **)malloc(self->n * sizeof(float *));
  safe_malloc(self->labels);

  /* Generate some random data. */
  for (size_t i = 0; i < self->n; ++i) {
    self->data[i] = (float *)malloc(self->lengths[i] * sizeof(float));
    safe_malloc(self->data[i]);
    for (size_t j = 0; j < self->lengths[i]; ++j) {
      float x = (float)rand() / (float)(RAND_MAX);
      self->data[i][j] = x;
    }

    self->labels[i] = (float *)malloc(self->lengths[i] * sizeof(float));
    safe_malloc(self->labels[i]);
    for (size_t j = 0; j < self->lengths[i]; ++j) {
      float y = (float)rand() / (float)(RAND_MAX);
      self->labels[i][j] = y;
    }
  }

  self->cur_it = 0;
  safe_xgboost(XGProxyDMatrixCreate(&self->_proxy));
}

void DataIterator_Free(DataIter *self) {
  for (size_t i = 0; i < self->n; ++i) {
    free(self->data[i]);
    free(self->labels[i]);
  }
  free(self->data);
  free(self->lengths);
  free(self->labels);
  safe_xgboost(XGDMatrixFree(self->_proxy));
};

int DataIterator_Next(DataIterHandle handle) {
  DataIter *self = (DataIter *)(handle);
  if (self->cur_it == self->n) {
    self->cur_it = 0;
    return 0;  /* At end */
  }

  /* A JSON string encoding array interface (standard from numpy). */
  char array[] = "{\"data\": [%lu, false], \"shape\":[%lu, 1], \"typestr\": "
                 "\"<f4\", \"version\": 3}";
  memset(self->_array, '\0', sizeof(self->_array));
  sprintf(self->_array, array, (size_t)self->data[self->cur_it],
          self->lengths[self->cur_it]);

  safe_xgboost(XGProxyDMatrixSetDataDense(self->_proxy, self->_array));
  /* The data passed in the iterator must remain valid (not being freed until the next
   * iteration or reset) */
  safe_xgboost(XGDMatrixSetDenseInfo(self->_proxy, "label",
                                     self->labels[self->cur_it],
                                     self->lengths[self->cur_it], 1));
  self->cur_it++;
  return 1;  /* Continue. */
}

void DataIterator_Reset(DataIterHandle handle) {
  DataIter *self = (DataIter *)(handle);
  self->cur_it = 0;
}

/**
 * Train a regression model and save it into JSON model file.
 */
void TrainModel(DMatrix Xy) {
  /* Create booster for training. */
  Booster booster;
  DMatrix cache[] = {Xy};
  safe_xgboost(XGBoosterCreate(cache, 1, &booster));
  /* Use approx or hist for external memory training. */
  safe_xgboost(XGBoosterSetParam(booster, "tree_method", "hist"));
  safe_xgboost(XGBoosterSetParam(booster, "objective", "reg:squarederror"));

  /* Start training. */
  const char *validation_names[1] = {"train"};
  const char *validation_result = NULL;
  size_t n_rounds = 10;
  for (size_t i = 0; i < n_rounds; ++i) {
    safe_xgboost(XGBoosterUpdateOneIter(booster, i, Xy));
    safe_xgboost(XGBoosterEvalOneIter(booster, i, cache, validation_names, 1,
                                      &validation_result));
    printf("%s\n", validation_result);
  }

  /* Save the model to a JSON file. */
  safe_xgboost(XGBoosterSaveModel(booster, "model.json"));

  safe_xgboost(XGBoosterFree(booster));
}

int main() {
  DataIter iter;
  DataIterator_Init(&iter, BATCH_LEN, N_BATCHS);

  /* Create DMatrix from iterator.  During training, some cache files with the
   * prefix "cache-" will be generated in current directory */
  char config[] = "{\"missing\": NaN, \"cache_prefix\": \"cache\"}";
  DMatrix Xy;
  safe_xgboost(XGDMatrixCreateFromCallback(
      &iter, iter._proxy, DataIterator_Reset, DataIterator_Next, config, &Xy));

  TrainModel(Xy);

  safe_xgboost(XGDMatrixFree(Xy));

  DataIterator_Free(&iter);
  return 0;
}
