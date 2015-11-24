/*!
 *  Copyright (c) 2014 by Contributors
 * \file page_dmatrix-inl.hpp
 *   row iterator based on sparse page
 * \author Tianqi Chen
 */
#ifndef XGBOOST_IO_PAGE_DMATRIX_INL_HPP_
#define XGBOOST_IO_PAGE_DMATRIX_INL_HPP_

#include <vector>
#include <string>
#include <algorithm>
#include "../data.h"
#include "../utils/iterator.h"
#include "../utils/thread_buffer.h"
#include "./simple_fmatrix-inl.hpp"
#include "./sparse_batch_page.h"
#include "./page_fmatrix-inl.hpp"
#include "./libsvm_parser.h"

namespace xgboost {
namespace io {
/*! \brief thread buffer iterator */
class ThreadRowPageIterator: public utils::IIterator<RowBatch> {
 public:
  ThreadRowPageIterator(void) {
    itr.SetParam("buffer_size", "4");
    page_ = NULL;
    base_rowid_ = 0;
  }
  virtual ~ThreadRowPageIterator(void) {}
  virtual void Init(void) {
  }
  virtual void BeforeFirst(void) {
    itr.BeforeFirst();
    base_rowid_ = 0;
  }
  virtual bool Next(void) {
    if (!itr.Next(page_)) return false;
    out_ = page_->GetRowBatch(base_rowid_);
    base_rowid_ += out_.size;
    return true;
  }
  virtual const RowBatch &Value(void) const {
    return out_;
  }
  /*! \brief load and initialize the iterator with fi */
  inline void Load(const utils::FileStream &fi) {
    itr.get_factory().SetFile(fi, 0);
    itr.Init();
    this->BeforeFirst();
  }

 private:
  // base row id
  size_t base_rowid_;
  // output data
  RowBatch out_;
  SparsePage *page_;
  utils::ThreadBuffer<SparsePage*, SparsePageFactory> itr;
};

/*! \brief data matrix using page */
template<int TKMagic>
class DMatrixPageBase : public DataMatrix {
 public:
  DMatrixPageBase(void) : DataMatrix(kMagic) {
    iter_ = new ThreadRowPageIterator();
  }
  // virtual destructor
  virtual ~DMatrixPageBase(void) {
    // do not delete row iterator, since it is owned by fmat
    // to be cleaned up in a more clear way
  }
  /*! \brief save a DataMatrix as DMatrixPage */
  inline static void Save(const char *fname_, const DataMatrix &mat, bool silent) {
    std::string fname = fname_;
    utils::FileStream fs(utils::FopenCheck(fname.c_str(), "wb"));
    int magic = kMagic;
    fs.Write(&magic, sizeof(magic));
    mat.info.SaveBinary(fs);
    fs.Close();
    fname += ".row.blob";
    utils::IIterator<RowBatch> *iter = mat.fmat()->RowIterator();
    utils::FileStream fbin(utils::FopenCheck(fname.c_str(), "wb"));
    SparsePage page;
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch &batch = iter->Value();
      for (size_t i = 0; i < batch.size; ++i) {
        page.Push(batch[i]);
        if (page.MemCostBytes() >= kPageSize) {
          page.Save(&fbin); page.Clear();
        }
      }
    }
    if (page.data.size() != 0) page.Save(&fbin);
    fbin.Close();
    if (!silent) {
      utils::Printf("DMatrixPage: %lux%lu is saved to %s\n",
                    static_cast<unsigned long>(mat.info.num_row()), // NOLINT(*)
                    static_cast<unsigned long>(mat.info.num_col()), fname_); // NOLINT(*)
    }
  }
  /*! \brief load and initialize the iterator with fi */
  inline void LoadBinary(utils::FileStream &fi,  // NOLINT(*)
                         bool silent,
                         const char *fname_) {
    this->set_cache_file(fname_);
    std::string fname = fname_;
    int tmagic;
    utils::Check(fi.Read(&tmagic, sizeof(tmagic)) != 0, "invalid input file format");
    this->CheckMagic(tmagic);
    this->info.LoadBinary(fi);
    // load in the row data file
    fname += ".row.blob";
    utils::FileStream fs(utils::FopenCheck(fname.c_str(), "rb"));
    iter_->Load(fs);
    if (!silent) {
      utils::Printf("DMatrixPage: %lux%lu matrix is loaded",
                    static_cast<unsigned long>(info.num_row()),  // NOLINT(*)
                    static_cast<unsigned long>(info.num_col()));  // NOLINT(*)
      if (fname_ != NULL) {
        utils::Printf(" from %s\n", fname_);
      } else {
        utils::Printf("\n");
      }
      if (info.group_ptr.size() != 0) {
        utils::Printf("data contains %u groups\n", (unsigned)info.group_ptr.size() - 1);
      }
    }
  }
  /*! \brief save a LibSVM format file as DMatrixPage */
  inline void LoadText(const char *uri,
                       const char* cache_file,
                       bool silent,
                       bool loadsplit) {
    if (!silent) {
      utils::Printf("start generate text file from %s\n", uri);
    }
    int rank = 0, npart = 1;
    if (loadsplit) {
      rank = rabit::GetRank();
      npart = rabit::GetWorldSize();
    }
    this->set_cache_file(cache_file);
    std::string fname_row = std::string(cache_file) + ".row.blob";
    utils::FileStream fo(utils::FopenCheck(fname_row.c_str(), "wb"));
    SparsePage page;
    size_t bytes_write = 0;
    double tstart = rabit::utils::GetTime();
    LibSVMParser parser(
        dmlc::InputSplit::Create(uri, rank, npart, "text"), 16);
    info.Clear();
    while (parser.Next()) {
      const LibSVMPage &batch = parser.Value();
      size_t nlabel = info.labels.size();
      info.labels.resize(nlabel + batch.label.size());
      if (batch.label.size() != 0) {
        std::memcpy(BeginPtr(info.labels) + nlabel,
                    BeginPtr(batch.label),
                    batch.label.size() * sizeof(float));
      }
      page.Push(batch);
      for (size_t i = 0; i < batch.data.size(); ++i) {
        info.info.num_col = std::max(info.info.num_col,
                                     static_cast<size_t>(batch.data[i].index+1));
      }
      if (page.MemCostBytes() >= kPageSize) {
        bytes_write += page.MemCostBytes();
        page.Save(&fo);
        page.Clear();
        double tdiff = rabit::utils::GetTime() - tstart;
        if (!silent) {
          utils::Printf("Writting to %s in %g MB/s, %lu MB written\n",
                        cache_file, (bytes_write >> 20UL) / tdiff,
                        (bytes_write >> 20UL));
        }
      }
      info.info.num_row += batch.label.size();
    }
    if (page.data.size() != 0) {
      page.Save(&fo);
    }
    fo.Close();
    iter_->Load(utils::FileStream(utils::FopenCheck(fname_row.c_str(), "rb")));
    // save data matrix
    utils::FileStream fs(utils::FopenCheck(cache_file, "wb"));
    int tmagic = kMagic;
    fs.Write(&tmagic, sizeof(tmagic));
    this->info.SaveBinary(fs);
    fs.Close();
    if (!silent) {
      utils::Printf("DMatrixPage: %lux%lu is parsed from %s\n",
                    static_cast<unsigned long>(info.num_row()),  // NOLINT(*)
                    static_cast<unsigned long>(info.num_col()),  // NOLINT(*)
                    uri);
    }
  }
  /*! \brief magic number used to identify DMatrix */
  static const int kMagic = TKMagic;
  /*! \brief page size 32 MB */
  static const size_t kPageSize = 32UL << 20UL;

 protected:
  virtual void set_cache_file(const std::string &cache_file)  = 0;
  virtual void CheckMagic(int tmagic)  = 0;
  /*! \brief row iterator */
  ThreadRowPageIterator *iter_;
};

class DMatrixPage : public DMatrixPageBase<0xffffab02> {
 public:
  DMatrixPage(void) {
    fmat_ = new FMatrixPage(iter_, this->info);
  }
  virtual ~DMatrixPage(void) {
    delete fmat_;
  }
  virtual IFMatrix *fmat(void) const {
    return fmat_;
  }
  virtual void set_cache_file(const std::string &cache_file) {
    fmat_->set_cache_file(cache_file);
  }
  virtual void CheckMagic(int tmagic) {
    utils::Check(tmagic == DMatrixPageBase<0xffffab02>::kMagic ||
                 tmagic == DMatrixPageBase<0xffffab03>::kMagic,
                 "invalid format,magic number mismatch");
  }
  /*! \brief the real fmatrix */
  FMatrixPage *fmat_;
};

// mix of FMatrix S and DMatrix
// cost half of ram usually as DMatrixSimple
class DMatrixHalfRAM : public DMatrixPageBase<0xffffab03> {
 public:
  DMatrixHalfRAM(void) {
    fmat_ = new FMatrixS(iter_, this->info);
  }
  virtual ~DMatrixHalfRAM(void) {
    delete fmat_;
  }
  virtual IFMatrix *fmat(void) const {
    return fmat_;
  }
  virtual void set_cache_file(const std::string &cache_file) {
  }
  virtual void CheckMagic(int tmagic) {
    utils::Check(tmagic == DMatrixPageBase<0xffffab02>::kMagic ||
                 tmagic == DMatrixPageBase<0xffffab03>::kMagic,
                 "invalid format,magic number mismatch");
  }
  /*! \brief the real fmatrix */
  IFMatrix *fmat_;
};
}  // namespace io
}  // namespace xgboost
#endif  // XGBOOST_IO_PAGE_ROW_ITER_INL_HPP_
