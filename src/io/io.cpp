#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#include <string>
#include "./io.h"
#include "simple_dmatrix-inl.hpp"
// implements data loads using dmatrix simple for now

namespace xgboost {
namespace io {
DataMatrix* LoadDataMatrix(const char *fname, bool silent, bool savebuffer) {
  DMatrixSimple *dmat = new DMatrixSimple();
  dmat->CacheLoad(fname, silent, savebuffer);
  return dmat;
}
}  // namespace io
}  // namespace xgboost
