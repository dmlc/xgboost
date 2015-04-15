#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX
#include <string>
#include "./io.h"
#include "../utils/io.h"
#include "../utils/utils.h"
#include "simple_dmatrix-inl.hpp"
// implements data loads using dmatrix simple for now

namespace xgboost {
namespace io {
DataMatrix* LoadDataMatrix(const char *fname, bool silent,
                           bool savebuffer, bool loadsplit) {
  if (!std::strcmp(fname, "stdin") ||
      !std::strncmp(fname, "s3://", 5) ||
      !std::strncmp(fname, "hdfs://", 7) ||
      loadsplit) {
    DMatrixSimple *dmat = new DMatrixSimple();
    dmat->LoadText(fname, silent, loadsplit);
    return dmat;
  }
  int magic;
  utils::FileStream fs(utils::FopenCheck(fname, "rb"));
  utils::Check(fs.Read(&magic, sizeof(magic)) != 0, "invalid input file format");
  fs.Seek(0);
  if (magic == DMatrixSimple::kMagic) { 
    DMatrixSimple *dmat = new DMatrixSimple();
    dmat->LoadBinary(fs, silent, fname);
    fs.Close();
    return dmat;
  }
  fs.Close();
  DMatrixSimple *dmat = new DMatrixSimple();
  dmat->CacheLoad(fname, silent, savebuffer);
  return dmat;
}

void SaveDataMatrix(const DataMatrix &dmat, const char *fname, bool silent) {
  if (dmat.magic == DMatrixSimple::kMagic) {
    const DMatrixSimple *p_dmat = static_cast<const DMatrixSimple*>(&dmat);
    p_dmat->SaveBinary(fname, silent);
  } else {
    DMatrixSimple smat;
    smat.CopyFrom(dmat);
    smat.SaveBinary(fname, silent);
  }
}

}  // namespace io
}  // namespace xgboost
