#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX
#include <string>
#include "./io.h"
#include "../utils/io.h"
#include "../utils/utils.h"
#include "simple_dmatrix-inl.hpp"
#ifndef XGBOOST_STRICT_CXX98_
#include "page_dmatrix-inl.hpp"
#include "page_fmatrix-inl.hpp"
#endif
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
#ifndef XGBOOST_STRICT_CXX98_
  std::string tmp_fname;
  const char *fname_ext = NULL;
  if (strchr(fname, ';') != NULL) {
    tmp_fname = fname;
    char *ptr = strchr(&tmp_fname[0], ';');
    ptr[0] = '\0'; fname = &tmp_fname[0];
    fname_ext = ptr + 1;
  }
  if (magic == DMatrixPage::kMagic) {
    if (fname_ext == NULL) {
      DMatrixPage *dmat = new DMatrixPage();
      dmat->Load(fs, silent, fname);
      return dmat;
    } else {
      DMatrixColPage *dmat = new DMatrixColPage(fname_ext);
      dmat->Load(fs, silent, fname, true);
      return dmat;
    }
  }
  if (magic == DMatrixColPage::kMagic) {
    std::string sfname = fname;
    if (fname_ext == NULL) {
      sfname += ".col"; fname_ext = sfname.c_str();
    }
    DMatrixColPage *dmat = new DMatrixColPage(fname_ext);
    dmat->Load(fs, silent, fname);
    return dmat;
  }
 #endif
  fs.Close();
  DMatrixSimple *dmat = new DMatrixSimple();
  dmat->CacheLoad(fname, silent, savebuffer);
  return dmat;
}

void SaveDataMatrix(const DataMatrix &dmat, const char *fname, bool silent) {
#ifndef XGBOOST_STRICT_CXX98_
  if (!strcmp(fname + strlen(fname) - 5, ".page")) {    
    DMatrixPage::Save(fname, dmat, silent);
    return;
  }
  if (!strcmp(fname + strlen(fname) - 6, ".cpage")) {
    DMatrixColPage::Save(fname, dmat, silent);
    return;
  }
#endif
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
