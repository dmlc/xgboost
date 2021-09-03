/*!
 * Copyright 2021 XGBoost contributors
 *
 * \brief A simple example of using prediction functions.
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

#define safe_malloc(ptr)                                                       \
  if ((ptr) == NULL) {                                                         \
    fprintf(stderr, "%s:%d: Failed to allocate memory.\n", __FILE__,           \
            __LINE__);                                                         \
    exit(1);                                                                   \
  }

#define N_SAMPLES 128
#define N_FEATURES 16

typedef BoosterHandle Booster;
typedef DMatrixHandle DMatrix;

/* Row-major matrix */
struct _Matrix {
  float *data;
  size_t shape[2];

  /* private members */
  char _array_intrerface[256];
};

/* A custom data type for demo. */
typedef struct _Matrix *Matrix;

/* Initialize matrix, copy data from `data` if it's not NULL. */
void Matrix_Create(Matrix *self, float const *data, size_t n_samples,
                   size_t n_features) {
  if (self == NULL) {
    fprintf(stderr, "Invalid pointer to %s\n", __func__);
    exit(-1);
  }

  *self = (Matrix)malloc(sizeof(struct _Matrix));
  safe_malloc(*self);
  (*self)->data = (float *)malloc(n_samples * n_features * sizeof(float));
  safe_malloc((*self)->data);
  (*self)->shape[0] = n_samples;
  (*self)->shape[1] = n_features;

  if (data != NULL) {
    memcpy((*self)->data, data,
           (*self)->shape[0] * (*self)->shape[1] * sizeof(float));
  }
}

/* Generate random matrix. */
void Matrix_Random(Matrix *self, size_t n_samples, size_t n_features) {
  Matrix_Create(self, NULL, n_samples, n_features);
  for (size_t i = 0; i < n_samples * n_features; ++i) {
    float x = (float)rand() / (float)(RAND_MAX);
    (*self)->data[i] = x;
  }
}

/* Array interface specified by numpy. */
char const *Matrix_ArrayInterface(Matrix self) {
  char const template[] = "{\"data\": [%lu, true], \"shape\": [%lu, %lu], "
                          "\"typestr\": \"<f4\", \"version\": 3}";
  memset(self->_array_intrerface, '\0', sizeof(self->_array_intrerface));
  sprintf(self->_array_intrerface, template, (size_t)self->data, self->shape[0],
          self->shape[1]);
  return self->_array_intrerface;
}

size_t Matrix_NSamples(Matrix self) { return self->shape[0]; }

size_t Matrix_NFeatures(Matrix self) { return self->shape[1]; }

float Matrix_At(Matrix self, size_t i, size_t j) {
  return self->data[i * self->shape[1] + j];
}

void Matrix_Print(Matrix self) {
  for (size_t i = 0; i < Matrix_NSamples(self); i++) {
    for (size_t j = 0; j < Matrix_NFeatures(self); ++j) {
      printf("%f, ", Matrix_At(self, i, j));
    }
  }
  printf("\n");
}

void Matrix_Free(Matrix self) {
  if (self != NULL) {
    if (self->data != NULL) {
      self->shape[0] = 0;
      self->shape[1] = 0;
      free(self->data);
      self->data = NULL;
    }
    free(self);
  }
}

int main() {
  Matrix X;
  Matrix y;

  Matrix_Random(&X, N_SAMPLES, N_FEATURES);
  Matrix_Random(&y, N_SAMPLES, 1);

  char const *X_interface = Matrix_ArrayInterface(X);
  char config[] = "{\"nthread\": 16, \"missing\": NaN}";
  DMatrix Xy;
  /* Dense means "dense matrix". */
  safe_xgboost(XGDMatrixCreateFromDense(X_interface, config, &Xy));
  /* Label must be in a contigious array. */
  safe_xgboost(XGDMatrixSetDenseInfo(Xy, "label", y->data, y->shape[0], 1));

  DMatrix cache[] = {Xy};
  Booster booster;
  /* Train a booster for demo. */
  safe_xgboost(XGBoosterCreate(cache, 1, &booster));

  size_t n_rounds = 10;
  for (size_t i = 0; i < n_rounds; ++i) {
    safe_xgboost(XGBoosterUpdateOneIter(booster, i, Xy));
  }

  /* Save the trained model in JSON format. */
  safe_xgboost(XGBoosterSaveModel(booster, "model.json"));
  safe_xgboost(XGBoosterFree(booster));

  /* Load it back for inference.  The save and load is not required, only shown here for
   * demonstration purpose. */
  safe_xgboost(XGBoosterCreate(NULL, 0, &booster));
  safe_xgboost(XGBoosterLoadModel(booster, "model.json"));
  {
    /* Run prediction with DMatrix object. */
    char const config[] =
        "{\"training\": false, \"type\": 0, "
        "\"iteration_begin\": 0, \"iteration_end\": 0, \"strict_shape\": true}";
    /* Shape of output prediction */
    uint64_t const *out_shape;
    /* Dimension of output prediction */
    uint64_t out_dim;
    /* Pointer to a thread local contigious array, assigned in prediction function. */
    float const *out_results;

    safe_xgboost(XGBoosterPredictFromDMatrix(booster, Xy, config, &out_shape,
                                             &out_dim, &out_results));
    if (out_dim != 2 || out_shape[0] != N_SAMPLES || out_shape[1] != 1) {
      fprintf(stderr, "Regression model should output prediction as vector.");
      exit(-1);
    }

    Matrix predt;
    /* Always copy output from XGBoost before calling next API function. */
    Matrix_Create(&predt, out_results, out_shape[0], out_shape[1]);
    printf("Results from prediction\n");
    Matrix_Print(predt);
    Matrix_Free(predt);
  }

  {
    /* Run inplace prediction, which is faster and more memory efficient, but supports
     * only basic inference types. */
    char const config[] = "{\"type\": 0, \"iteration_begin\": 0, "
                          "\"iteration_end\": 0, \"strict_shape\": true, "
                          "\"cache_id\": 0, \"missing\": NaN}";
    /* Shape of output prediction */
    uint64_t const *out_shape;
    /* Dimension of output prediction */
    uint64_t out_dim;
    /* Pointer to a thread local contigious array, assigned in prediction function. */
    float const *out_results;

    char const *X_interface = Matrix_ArrayInterface(X);
    safe_xgboost(XGBoosterPredictFromDense(booster, X_interface, config, NULL,
                                           &out_shape, &out_dim, &out_results));

    if (out_dim != 2 || out_shape[0] != N_SAMPLES || out_shape[1] != 1) {
      fprintf(stderr,
              "Regression model should output prediction as vector, %lu, %lu",
              out_dim, out_shape[0]);
      exit(-1);
    }

    Matrix predt;
    /* Always copy output from XGBoost before calling next API function. */
    Matrix_Create(&predt, out_results, out_shape[0], out_shape[1]);
    printf("Results from inplace prediction\n");
    Matrix_Print(predt);
    Matrix_Free(predt);
  }

  XGBoosterFree(booster);

  XGDMatrixFree(Xy);
  Matrix_Free(X);
  Matrix_Free(y);
  return 0;
}
