#ifndef XGBOOST_IO_SIMPLE_DMATRIX_INL_HPP_
#define XGBOOST_IO_SIMPLE_DMATRIX_INL_HPP_
/*!
 * \file simple_dmatrix-inl.hpp
 * \brief simple implementation of DMatrixS that can be used 
 *  the data format of xgboost is templatized, which means it can accept
 *  any data structure that implements the function defined by FMatrix
 *  this file is a specific implementation of input data structure that can be used by BoostLearner
 * \author Tianqi Chen
 */
#include <string>
#include <cstring>
#include <vector>
#include <algorithm>
#include "../data.h"
#include "../utils/utils.h"
#include "../learner/dmatrix.h"
#include "./io.h"

namespace xgboost {
namespace io {
/*! \brief implementation of DataMatrix, in CSR format */
class DMatrixSimple : public DataMatrix {
 public:
  // constructor
  DMatrixSimple(void) : DataMatrix(kMagic) {
    this->fmat.set_iter(new OneBatchIter(this));
    this->Clear();
  }
  // virtual destructor
  virtual ~DMatrixSimple(void) {}
  /*! \brief clear the storage */
  inline void Clear(void) {
    row_ptr_.clear();
    row_ptr_.push_back(0);
    row_data_.clear();
    info.Clear();
  }
  /*! \brief copy content data from source matrix */
  inline void CopyFrom(const DataMatrix &src) {
    this->info = src.info;
    this->Clear();
    // clone data content in thos matrix
    utils::IIterator<SparseBatch> *iter = src.fmat.RowIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      const SparseBatch &batch = iter->Value();
      for (size_t i = 0; i < batch.size; ++i) {
        SparseBatch::Inst inst = batch[i];
        row_data_.resize(row_data_.size() + inst.length);
        memcpy(&row_data_[row_ptr_.back()], inst.data,
               sizeof(SparseBatch::Entry) * inst.length);
        row_ptr_.push_back(row_ptr_.back() + inst.length);
      }
    }
  }
  /*!
   * \brief add a row to the matrix
   * \param feats features
   * \return the index of added row
   */
  inline size_t AddRow(const std::vector<SparseBatch::Entry> &feats) {
    for (size_t i = 0; i < feats.size(); ++i) {
      row_data_.push_back(feats[i]);
      info.num_col = std::max(info.num_col, static_cast<size_t>(feats[i].findex+1));
    }
    row_ptr_.push_back(row_ptr_.back() + feats.size());
    info.num_row += 1;
    return row_ptr_.size() - 2;
  }
  /*!
   * \brief load from text file
   * \param fname name of text data
   * \param silent whether print information or not
   */
  inline void LoadText(const char* fname, bool silent = false) {
    this->Clear();
    FILE* file = utils::FopenCheck(fname, "r");
    float label; bool init = true;
    char tmp[1024];
    std::vector<SparseBatch::Entry> feats;
    while (fscanf(file, "%s", tmp) == 1) {
      SparseBatch::Entry e;
      if (sscanf(tmp, "%u:%f", &e.findex, &e.fvalue) == 2) {
        feats.push_back(e);
      } else {
        if (!init) {
          info.labels.push_back(label);
          this->AddRow(feats);
        }
        feats.clear();
        utils::Check(sscanf(tmp, "%f", &label) == 1, "invalid LibSVM format");
        init = false;
      }
    }

    info.labels.push_back(label);
    this->AddRow(feats);

    if (!silent) {
      printf("%lux%lu matrix with %lu entries is loaded from %s\n",
             info.num_row, info.num_col, row_data_.size(), fname);
    }
    fclose(file);
    // try to load in additional file
    std::string name = fname;
    std::string gname = name + ".group";
    if (info.TryLoadGroup(gname.c_str(), silent)) {
      utils::Check(info.group_ptr.back() == info.num_row,
                   "DMatrix: group data does not match the number of rows in features");
    }
    std::string wname = name + ".weight";
    if (info.TryLoadWeight(wname.c_str(), silent)) {
      utils::Check(info.weights.size() == info.num_row,
                   "DMatrix: weight data does not match the number of rows in features");
    }
  }
  /*!
   * \brief load from binary file
   * \param fname name of binary data
   * \param silent whether print information or not
   * \return whether loading is success
   */
  inline bool LoadBinary(const char* fname, bool silent = false) {
    FILE *fp = fopen64(fname, "rb");
    if (fp == NULL) return false;
    utils::FileStream fs(fp);
    int magic;
    utils::Check(fs.Read(&magic, sizeof(magic)) != 0, "invalid input file format");
    utils::Check(magic == kMagic, "invalid format,magic number mismatch");

    info.LoadBinary(fs);
    FMatrixS::LoadBinary(fs, &row_ptr_, &row_data_);
    fmat.LoadColAccess(fs);
    fs.Close();

    if (!silent) {
      printf("%lux%lu matrix with %lu entries is loaded from %s\n",
             info.num_row, info.num_col, row_data_.size(), fname);
      if (info.group_ptr.size() != 0) {
        printf("data contains %u groups\n", (unsigned)info.group_ptr.size()-1);
      }
    }
    return true;
  }
  /*!
   * \brief save to binary file
   * \param fname name of binary data
   * \param silent whether print information or not
   */
  inline void SaveBinary(const char* fname, bool silent = false) const {
    utils::FileStream fs(utils::FopenCheck(fname, "wb"));
    int magic = kMagic;
    fs.Write(&magic, sizeof(magic));

    info.SaveBinary(fs);
    FMatrixS::SaveBinary(fs, row_ptr_, row_data_);
    fmat.SaveColAccess(fs);
    fs.Close();

    if (!silent) {
      printf("%lux%lu matrix with %lu entries is saved to %s\n",
             info.num_row, info.num_col, row_data_.size(), fname);
      if (info.group_ptr.size() != 0) {
        printf("data contains %lu groups\n", info.group_ptr.size()-1);
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
    int len = strlen(fname);
    if (len > 8 && !strcmp(fname + len - 7, ".buffer")) {
      if (!this->LoadBinary(fname, silent)) {
        utils::Error("can not open file \"%s\"", fname);
      }
      return;
    }
    char bname[1024];
    snprintf(bname, sizeof(bname), "%s.buffer", fname);
    if (!this->LoadBinary(bname, silent)) {
      this->LoadText(fname, silent);
      if (savebuffer) this->SaveBinary(bname, silent);
    }
  }
  // data fields
  /*! \brief row pointer of CSR sparse storage */
  std::vector<size_t> row_ptr_;
  /*! \brief data in the row */
  std::vector<SparseBatch::Entry> row_data_;
  /*! \brief magic number used to identify DMatrix */
  static const int kMagic = 0xffffab01;

 protected:
  // one batch iterator that return content in the matrix
  struct OneBatchIter: utils::IIterator<SparseBatch> {
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
      batch_.row_ptr = &parent_->row_ptr_[0];
      batch_.data_ptr = &parent_->row_data_[0];
      return true;
    }
    virtual const SparseBatch &Value(void) const {
      return batch_;
    }

   private:
    // whether is at first
    bool at_first_;
    // pointer to parient
    DMatrixSimple *parent_;
    // temporal space for batch
    SparseBatch batch_;
  };
};
}  // namespace io
}  // namespace xgboost
#endif  // namespace XGBOOST_IO_SIMPLE_DMATRIX_INL_HPP_
