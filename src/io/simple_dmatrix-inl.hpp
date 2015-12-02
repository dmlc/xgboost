/*!
 * Copyright 2014 by Contributors
 * \file simple_dmatrix-inl.hpp
 * \brief simple implementation of DMatrixS that can be used
 *  the data format of xgboost is templatized, which means it can accept
 *  any data structure that implements the function defined by FMatrix
 *  this file is a specific implementation of input data structure that can be used by BoostLearner
 * \author Tianqi Chen
 */
#ifndef XGBOOST_IO_SIMPLE_DMATRIX_INL_HPP_
#define XGBOOST_IO_SIMPLE_DMATRIX_INL_HPP_

#include <string>
#include <cstring>
#include <vector>
#include <sstream>
#include <algorithm>
#include "../data.h"
#include "../utils/utils.h"
#include "../learner/dmatrix.h"
#include "./io.h"
#include "./simple_fmatrix-inl.hpp"
#include "../sync/sync.h"
#include "./libsvm_parser.h"

namespace xgboost {
namespace io {
/*! \brief implementation of DataMatrix, in CSR format */
class DMatrixSimple : public DataMatrix {
 public:
  // constructor
  DMatrixSimple(void) : DataMatrix(kMagic) {
    fmat_ = new FMatrixS(new OneBatchIter(this), this->info);
    this->Clear();
  }
  // virtual destructor
  virtual ~DMatrixSimple(void) {
    delete fmat_;
  }
  virtual IFMatrix *fmat(void) const {
    return fmat_;
  }
  /*! \brief clear the storage */
  inline void Clear(void) {
    row_ptr_.clear();
    row_ptr_.push_back(0);
    row_data_.clear();
    info.Clear();
  }
  /*! \brief copy content data from source matrix */
  inline void CopyFrom(const DataMatrix &src) {
    this->Clear();
    this->info = src.info;
    // clone data content in thos matrix
    utils::IIterator<RowBatch> *iter = src.fmat()->RowIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch &batch = iter->Value();
      for (size_t i = 0; i < batch.size; ++i) {
        RowBatch::Inst inst = batch[i];
        row_data_.resize(row_data_.size() + inst.length);
        if (inst.length != 0) {
          std::memcpy(&row_data_[row_ptr_.back()], inst.data,
                      sizeof(RowBatch::Entry) * inst.length);
        }
        row_ptr_.push_back(row_ptr_.back() + inst.length);
      }
    }
  }
  /*!
   * \brief add a row to the matrix
   * \param feats features
   * \return the index of added row
   */
  inline size_t AddRow(const std::vector<RowBatch::Entry> &feats) {
    for (size_t i = 0; i < feats.size(); ++i) {
      row_data_.push_back(feats[i]);
      info.info.num_col = std::max(info.info.num_col,
                                   static_cast<size_t>(feats[i].index+1));
    }
    row_ptr_.push_back(row_ptr_.back() + feats.size());
    info.info.num_row += 1;
    return row_ptr_.size() - 2;
  }
  /*!
   * \brief load split of input, used in distributed mode
   * \param uri the uri of input
   * \param loadsplit whether loadsplit of data or all the data
   * \param silent whether print information or not
   */
  inline void LoadText(const char *uri, bool silent = false, bool loadsplit = false) {
    int rank = 0, npart = 1;
    if (loadsplit) {
      rank = rabit::GetRank();
      npart = rabit::GetWorldSize();
    }
    LibSVMParser parser(
        dmlc::InputSplit::Create(uri, rank, npart, "text"), 16);
    this->Clear();
    while (parser.Next()) {
      const LibSVMPage &batch = parser.Value();
      size_t nlabel = info.labels.size();
      info.labels.resize(nlabel + batch.label.size());
      if (batch.label.size() != 0) {
        std::memcpy(BeginPtr(info.labels) + nlabel,
                    BeginPtr(batch.label),
                    batch.label.size() * sizeof(float));
      }
      size_t ndata = row_data_.size();
      row_data_.resize(ndata + batch.data.size());
      if (batch.data.size() != 0) {
        std::memcpy(BeginPtr(row_data_) + ndata,
                    BeginPtr(batch.data),
                    batch.data.size() * sizeof(RowBatch::Entry));
      }
      row_ptr_.resize(row_ptr_.size() + batch.label.size());
      for (size_t i = 0; i < batch.label.size(); ++i) {
        row_ptr_[nlabel + i + 1] = row_ptr_[nlabel] + batch.offset[i + 1];
      }
      info.info.num_row += batch.Size();
      for (size_t i = 0; i < batch.data.size(); ++i) {
        info.info.num_col = std::max(info.info.num_col,
                                     static_cast<size_t>(batch.data[i].index+1));
      }
    }
    if (!silent) {
      utils::Printf("%lux%lu matrix with %lu entries is loaded from %s\n",
                    static_cast<unsigned long>(info.num_row()),  // NOLINT(*)
                    static_cast<unsigned long>(info.num_col()),  // NOLINT(*)
                    static_cast<unsigned long>(row_data_.size()), uri);  // NOLINT(*)
    }
    // try to load in additional file
    if (!loadsplit) {
      std::string name = uri;
      std::string gname = name + ".group";
      if (info.TryLoadGroup(gname.c_str(), silent)) {
        utils::Check(info.group_ptr.back() == info.num_row(),
                     "DMatrix: group data does not match the number of rows in features");
      }
      std::string wname = name + ".weight";
      if (info.TryLoadFloatInfo("weight", wname.c_str(), silent)) {
        utils::Check(info.weights.size() == info.num_row(),
                     "DMatrix: weight data does not match the number of rows in features");
      }
      std::string mname = name + ".base_margin";
      if (info.TryLoadFloatInfo("base_margin", mname.c_str(), silent)) {
      }
    }
  }
  /*!
   * \brief load from binary file
   * \param fname name of binary data
   * \param silent whether print information or not
   * \return whether loading is success
   */
  inline bool LoadBinary(const char* fname, bool silent = false) {
    std::FILE *fp = fopen64(fname, "rb");
    if (fp == NULL) return false;
    utils::FileStream fs(fp);
    this->LoadBinary(fs, silent, fname);
    fs.Close();
    return true;
  }
  /*!
   * \brief load from binary stream
   * \param fs input file stream
   * \param silent whether print information during loading
   * \param fname file name, used to print message
   */
  inline void LoadBinary(utils::IStream &fs, bool silent = false, const char *fname = NULL) {  // NOLINT(*)
    int tmagic;
    utils::Check(fs.Read(&tmagic, sizeof(tmagic)) != 0, "invalid input file format");
    utils::Check(tmagic == kMagic, "\"%s\" invalid format, magic number mismatch",
                 fname == NULL ? "" : fname);

    info.LoadBinary(fs);
    LoadBinary(fs, &row_ptr_, &row_data_);
    fmat_->LoadColAccess(fs);

    if (!silent) {
      utils::Printf("%lux%lu matrix with %lu entries is loaded",
                    static_cast<unsigned long>(info.num_row()),  // NOLINT(*)
                    static_cast<unsigned long>(info.num_col()),  // NOLINT(*)
                    static_cast<unsigned long>(row_data_.size()));  // NOLINT(*)
      if (fname != NULL) {
        utils::Printf(" from %s\n", fname);
      } else {
        utils::Printf("\n");
      }
      if (info.group_ptr.size() != 0) {
        utils::Printf("data contains %u groups\n", (unsigned)info.group_ptr.size()-1);
      }
    }
  }
  /*!
   * \brief save to binary file
   * \param fname name of binary data
   * \param silent whether print information or not
   */
  inline void SaveBinary(const char* fname, bool silent = false) const {
    utils::FileStream fs(utils::FopenCheck(fname, "wb"));
    int tmagic = kMagic;
    fs.Write(&tmagic, sizeof(tmagic));
    info.SaveBinary(fs);
    SaveBinary(fs, row_ptr_, row_data_);
    fmat_->SaveColAccess(fs);
    fs.Close();

    if (!silent) {
      utils::Printf("%lux%lu matrix with %lu entries is saved to %s\n",
                    static_cast<unsigned long>(info.num_row()),  // NOLINT(*)
                    static_cast<unsigned long>(info.num_col()),  // NOLINT(*)
                    static_cast<unsigned long>(row_data_.size()), fname);  // NOLINT(*)
      if (info.group_ptr.size() != 0) {
        utils::Printf("data contains %u groups\n",
                      static_cast<unsigned>(info.group_ptr.size()-1));
      }
    }
  }
  /*!
   * \brief cache load data given a file name, if filename ends with .buffer, direct load binary
   *        otherwise the function will first check if fname + '.buffer' exists,
   *        if binary buffer exists, it will reads from binary buffer, otherwise, it will load from text file,
   *        and try to create a buffer file
   * \param fname name of binary data
   * \param silent whether print information or not
   * \param savebuffer whether do save binary buffer if it is text
   */
  inline void CacheLoad(const char *fname, bool silent = false, bool savebuffer = true) {
    using namespace std;
    size_t len = strlen(fname);
    if (len > 8 && !strcmp(fname + len - 7, ".buffer")) {
      if (!this->LoadBinary(fname, silent)) {
        utils::Error("can not open file \"%s\"", fname);
      }
      return;
    }
    char bname[1024];
    utils::SPrintf(bname, sizeof(bname), "%s.buffer", fname);
    if (!this->LoadBinary(bname, silent)) {
      this->LoadText(fname, silent);
      if (savebuffer) this->SaveBinary(bname, silent);
    }
  }
  // data fields
  /*! \brief row pointer of CSR sparse storage */
  std::vector<size_t> row_ptr_;
  /*! \brief data in the row */
  std::vector<RowBatch::Entry> row_data_;
  /*! \brief the real fmatrix */
  FMatrixS *fmat_;
  /*! \brief magic number used to identify DMatrix */
  static const int kMagic = 0xffffab01;

 protected:
  /*!
   * \brief save data to binary stream
   * \param fo output stream
   * \param ptr pointer data
   * \param data data content
   */
  inline static void SaveBinary(utils::IStream &fo,  // NOLINT(*)
                                const std::vector<size_t> &ptr,
                                const std::vector<RowBatch::Entry> &data) {
    size_t nrow = ptr.size() - 1;
    fo.Write(&nrow, sizeof(size_t));
    fo.Write(BeginPtr(ptr), ptr.size() * sizeof(size_t));
    if (data.size() != 0) {
      fo.Write(BeginPtr(data), data.size() * sizeof(RowBatch::Entry));
    }
  }
  /*!
   * \brief load data from binary stream
   * \param fi input stream
   * \param out_ptr pointer data
   * \param out_data data content
   */
  inline static void LoadBinary(utils::IStream &fi,  // NOLINT(*)
                                std::vector<size_t> *out_ptr,
                                std::vector<RowBatch::Entry> *out_data) {
    size_t nrow;
    utils::Check(fi.Read(&nrow, sizeof(size_t)) != 0, "invalid input file format");
    out_ptr->resize(nrow + 1);
    utils::Check(fi.Read(BeginPtr(*out_ptr), out_ptr->size() * sizeof(size_t)) != 0,
                  "invalid input file format");
    out_data->resize(out_ptr->back());
    if (out_data->size() != 0) {
      utils::Assert(fi.Read(BeginPtr(*out_data), out_data->size() * sizeof(RowBatch::Entry)) != 0,
                    "invalid input file format");
    }
  }
  // one batch iterator that return content in the matrix
  struct OneBatchIter: utils::IIterator<RowBatch> {
    explicit OneBatchIter(DMatrixSimple *parent)
        : at_first_(true), parent_(parent) {}
    virtual ~OneBatchIter(void) {}
    virtual void BeforeFirst(void) {
      at_first_ = true;
    }
    virtual bool Next(void) {
      if (!at_first_) return false;
      at_first_ = false;
      batch_.size = parent_->row_ptr_.size() - 1;
      batch_.base_rowid = 0;
      batch_.ind_ptr = BeginPtr(parent_->row_ptr_);
      batch_.data_ptr = BeginPtr(parent_->row_data_);
      return true;
    }
    virtual const RowBatch &Value(void) const {
      return batch_;
    }

   private:
    // whether is at first
    bool at_first_;
    // pointer to parient
    DMatrixSimple *parent_;
    // temporal space for batch
    RowBatch batch_;
  };
};
}  // namespace io
}  // namespace xgboost
#endif  // namespace XGBOOST_IO_SIMPLE_DMATRIX_INL_HPP_
