/*!
 *  Copyright (c) 2015 by Contributors
 * \file toolkit_util.h
 * \brief simple data structure that could be used by model
 *
 * \author Tianqi Chen
 */
#ifndef RABIT_TOOLKIT_UTIL_H_
#define RABIT_TOOLKIT_UTIL_H_

#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <cmath>
#include <rabit.h>

namespace rabit {
// typedef index type
typedef unsigned index_t;

/*! \brief sparse matrix, CSR format */
struct SparseMat {
  // sparse matrix entry
  struct Entry {
    // feature index 
    index_t findex;
    // feature value
    float fvalue;
  };
  // sparse vector
  struct Vector {
    const Entry *data;
    index_t length;
    inline const Entry &operator[](size_t i) const {
      return data[i];
    }
  };
  inline Vector operator[](size_t i) const {
    Vector v;
    v.data = &data[0] + row_ptr[i];
    v.length = static_cast<index_t>(row_ptr[i + 1]-row_ptr[i]);
    return v;
  }
  // load data from LibSVM format
  inline void Load(const char *fname) {
    FILE *fi;
    if (!strcmp(fname, "stdin")) {
      fi = stdin;
    } else {
      if (strchr(fname, '%') != NULL) {
        char s_tmp[256];
        snprintf(s_tmp, sizeof(s_tmp), fname, rabit::GetRank());
        fi = utils::FopenCheck(s_tmp, "r");        
      } else {
        fi = utils::FopenCheck(fname, "r");
      }
    }
    row_ptr.clear();
    row_ptr.push_back(0);
    data.clear();    
    feat_dim = 0;
    float label; bool init = true;
    char tmp[1024];
    while (fscanf(fi, "%s", tmp) == 1) {
      Entry e;
      unsigned long fidx;
      if (sscanf(tmp, "%lu:%f", &fidx, &e.fvalue) == 2) {
        e.findex = static_cast<index_t>(fidx);
        data.push_back(e);
        feat_dim = std::max(fidx, feat_dim);
      } else {
        if (!init) {
          labels.push_back(label);
          row_ptr.push_back(data.size());
        }
        utils::Check(sscanf(tmp, "%f", &label) == 1, "invalid LibSVM format");
        init = false;
      }
    }
    // last row
    labels.push_back(label);
    row_ptr.push_back(data.size());
    feat_dim += 1;
    utils::Check(feat_dim < std::numeric_limits<index_t>::max(),
                 "feature dimension exceed limit of index_t"\
                 "consider change the index_t to unsigned long");
    // close the filed
    if (fi != stdin) fclose(fi);
  }
  inline size_t NumRow(void) const {
    return row_ptr.size() - 1;
  }
  // maximum feature dimension
  size_t feat_dim;
  std::vector<size_t> row_ptr;
  std::vector<Entry> data;
  std::vector<float> labels;
};
// dense matrix
struct Matrix {
  inline void Init(size_t nrow, size_t ncol, float v = 0.0f) {
    this->nrow = nrow;
    this->ncol = ncol;
    data.resize(nrow * ncol);
    std::fill(data.begin(), data.end(), v);
  }
  inline float *operator[](size_t i) {
    return &data[0] + i * ncol;
  }
  inline const float *operator[](size_t i) const {
    return &data[0] + i * ncol;
  }
  inline void Print(const char *fname) {
    FILE *fo;
    if (!strcmp(fname, "stdout")) {
      fo = stdout;
    } else {
      fo = utils::FopenCheck(fname, "w");
    }
    for (size_t i = 0; i < data.size(); ++i) {
      fprintf(fo, "%g", data[i]);
      if ((i+1) % ncol == 0) {
        fprintf(fo, "\n");
      } else {
        fprintf(fo, " ");
      }
    }
    // close the filed
    if (fo != stdout) fclose(fo);
  }
  // number of data
  size_t nrow, ncol;
  std::vector<float> data;
};

/*!\brief computes a random number modulo the value */
inline int Random(int value) {
  return rand() % value;
}
} // namespace rabit
#endif // RABIT_TOOLKIT_UTIL_H_
