/*!
 *  Copyright (c) 2015 by Contributors
 * \file data.h
 * \brief simple data structure that could be used by model
 *
 * \author Tianqi Chen
 */
#ifndef RABIT_LEARN_DATA_H_
#define RABIT_LEARN_DATA_H_

#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <cmath>
#include <sstream>
#include <rabit.h>
#include "../io/io.h"

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
    io::InputSplit *in =
        io::CreateInputSplit
        (fname, rabit::GetRank(),
         rabit::GetWorldSize());
    row_ptr.clear();
    row_ptr.push_back(0);
    data.clear();    
    feat_dim = 0;
    std::string line;
    while (in->NextLine(&line)) {
      float label;
      std::istringstream ss(line);
      ss >> label;
      Entry e;
      unsigned long fidx;
      while (!ss.eof()) {
        if (!(ss >> fidx)) break;
        ss.ignore(32, ':');
        if (!(ss >> e.fvalue)) break;
        e.findex = static_cast<index_t>(fidx);
        data.push_back(e);
        feat_dim = std::max(fidx, feat_dim);
      }
      labels.push_back(label);
      row_ptr.push_back(data.size());
    }
    delete in;
    feat_dim += 1;
    utils::Check(feat_dim < std::numeric_limits<index_t>::max(),
                 "feature dimension exceed limit of index_t"\
                 "consider change the index_t to unsigned long");
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
#endif // RABIT_LEARN_DATA_H_
