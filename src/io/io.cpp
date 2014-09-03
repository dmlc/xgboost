#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#include <string>
#include "./io.h"
#include "../utils/io.h"
#include "../utils/utils.h"
#include "simple_dmatrix-inl.hpp"
#include "page_dmatrix-inl.hpp"
#include "page_fmatrix-inl.hpp"

// implements data loads using dmatrix simple for now

namespace xgboost {
namespace io {
DataMatrix* LoadDataMatrix(const char *fname, bool silent, bool savebuffer) {
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
  if (magic == DMatrixPage::kMagic) {
    DMatrixPage *dmat = new DMatrixPage();
    dmat->Load(fs, silent, fname);
    // the file pointer is hold in page matrix
    return dmat;
  }
  if (magic == DMatrixColPage::kMagic) {
    DMatrixColPage *dmat = new DMatrixColPage(fname);
    dmat->Load(fs, silent, fname);
    // the file pointer is hold in page matrix
    return dmat;
  }
  fs.Close();
 
  DMatrixSimple *dmat = new DMatrixSimple();
  dmat->CacheLoad(fname, silent, savebuffer);
  return dmat;
}

void SaveDataMatrix(const DataMatrix &dmat, const char *fname, bool silent) {
  if (!strcmp(fname + strlen(fname) - 5, ".page")) {    
    DMatrixPage::Save(fname, dmat, silent);
    return;
  }
  if (!strcmp(fname + strlen(fname) - 6, ".cpage")) {
    DMatrixColPage::Save(fname, dmat, silent);
    return;
  }
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
