#ifndef XGBOOST_IO_PAGE_ROW_ITER_INL_HPP_
#define XGBOOST_IO_PAGE_ROW_ITER_INL_HPP_
/*!
 * \file page_row_iter-inl.hpp
 *   row iterator based on sparse page
 * \author Tianqi Chen
 */
#include <vector>
#include "../data.h"
#include "../utils/iterator.h"
#include "../utils/thread_buffer.h"
#include "./simple_fmatrix-inl.hpp"

namespace xgboost {
namespace io {
/*! \brief page structure that can be used to store a rowbatch */
struct RowBatchPage {
 public:
  explicit RowBatchPage(size_t page_size) : kPageSize(page_size) {
    data_ = new int[kPageSize];
    utils::Assert(data_ != NULL, "fail to allocate row batch page");
    this->Clear();
  }
  ~RowBatchPage(void) {
    if (data_ != NULL) delete [] data_;
  }
  /*! 
   * \brief Push one row into page
   *  \param row an instance row
   *  \return false or true to push into
   */  
  inline bool PushRow(const RowBatch::Inst &row) {
    const size_t dsize = row.length * sizeof(RowBatch::Entry);
    if (FreeBytes() < dsize+ sizeof(int)) return false;
    row_ptr(Size() + 1) = row_ptr(Size()) + row.length;
    memcpy(data_ptr(row_ptr(Size())) , row.data, dsize);
    ++data_[0];
    return true;
  }
  /*!
   * \brief get a row batch representation from the page
   * \param p_rptr a temporal space that can be used to provide
   *  ind_ptr storage for RowBatch
   * \return a new RowBatch object
   */
  inline RowBatch GetRowBatch(std::vector<size_t> *p_rptr, size_t base_rowid) {
    RowBatch batch;
    batch.base_rowid = base_rowid;
    batch.data_ptr = this->data_ptr(0);
    batch.size = static_cast<size_t>(this->Size());
    std::vector<size_t> &rptr = *p_rptr;
    rptr.resize(this->Size() + 1);
    for (size_t i = 0; i < rptr.size(); ++i) {
      rptr[i] = static_cast<size_t>(this->row_ptr(static_cast<int>(i)));
    }
    batch.ind_ptr = &rptr[0];
    return batch;
  }
  /*! \brief get i-th row from the batch */
  inline RowBatch::Inst operator[](int i) {
    return RowBatch::Inst(data_ptr(0) + row_ptr(i),
                          static_cast<bst_uint>(row_ptr(i+1) - row_ptr(i)));
  }
  /*!
   * \brief clear the page, cleanup the content
   */
  inline void Clear(void) {
    memset(&data_[0], 0, sizeof(int) * kPageSize);
  }
  /*!
   * \brief load one page form instream
   * \return true if loading is successful
   */
  inline bool Load(utils::IStream &fi) {
    return fi.Read(&data_[0], sizeof(int) * kPageSize) != 0;
  }
  /*! \brief save one page into outstream */
  inline void Save(utils::IStream &fo) {
    fo.Write(&data_[0], sizeof(int) * kPageSize);
  }
  /*! \return number of elements */
  inline int Size(void) const {
    return data_[0];
  }

 protected:
  /*! \return number of elements */
  inline size_t FreeBytes(void) {
    return (kPageSize - (Size() + 2)) * sizeof(int) -
        row_ptr(Size()) * sizeof(RowBatch::Entry);
  }
  /*! \brief equivalent row pointer at i */
  inline int& row_ptr(int i) {
    return data_[kPageSize - i - 1];
  }
  inline RowBatch::Entry* data_ptr(int i) {
    return (RowBatch::Entry*)(&data_[1]) + i;
  }
  // content of data
  int *data_;
  // page size
  const size_t kPageSize;
};
/*! \brief thread buffer iterator */
class ThreadRowPageIterator: public utils::IIterator<RowBatch> {
 public:
  ThreadRowPageIterator(void) {
    itr.SetParam("buffer_size", "2");
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
    out_ = page_->GetRowBatch(&tmp_ptr_, base_rowid_);
    base_rowid_ += out_.size;
    return true;
  }
  virtual const RowBatch &Value(void) const {
    return out_;
  }
  /*! \brief load and initialize the iterator with fi */
  inline void Load(const utils::FileStream &fi) {
    itr.get_factory().SetFile(fi);
    itr.Init();
    this->BeforeFirst();
  }
  /*!
   * \brief save a row iterator to output stream, in row iterator format
   */
  inline static void Save(utils::IIterator<RowBatch> *iter,
                          utils::IStream &fo) {
    RowBatchPage page(kPageSize);
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch &batch = iter->Value();
      for (size_t i = 0; i < batch.size; ++i) {
        if (!page.PushRow(batch[i])) {
          page.Save(fo);
          page.Clear();
          utils::Check(page.PushRow(batch[i]), "row is too big");
        }
      }
    }
    if (page.Size() != 0) page.Save(fo);
  }
  /*! \brief page size 64 MB */
  static const size_t kPageSize = 64 << 18;

 private:
  // base row id
  size_t base_rowid_;
  // temporal ptr
  std::vector<size_t> tmp_ptr_;
  // output data
  RowBatch out_;
  // page pointer type
  typedef RowBatchPage* PagePtr;
  // loader factory for page
  struct Factory {
   public:
    size_t file_begin_;
    utils::FileStream fi;
    Factory(void) {}
    inline void SetFile(const utils::FileStream &fi) {
      this->fi = fi;
      file_begin_ = this->fi.Tell();
    }
    inline bool Init(void) {
      return true;
    }
    inline void SetParam(const char *name, const char *val) {}
    inline bool LoadNext(PagePtr &val) {
      return val->Load(fi);
    }
    inline PagePtr Create(void) {
      PagePtr a = new RowBatchPage(kPageSize);
      return a;
    }
    inline void FreeSpace(PagePtr &a) {
      delete a;
    }
    inline void Destroy(void) {
      fi.Close();
    }
    inline void BeforeFirst(void) {
      fi.Seek(file_begin_);
    }
  };

 protected:
  PagePtr page_;
  utils::ThreadBuffer<PagePtr, Factory> itr;
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
  /*! \brief load and initialize the iterator with fi */
  inline void LoadBinary(utils::FileStream &fi,
                         bool silent,
                         const char *fname_) {
    std::string fname = fname_;
    int tmagic;
    utils::Check(fi.Read(&tmagic, sizeof(tmagic)) != 0, "invalid input file format");
    utils::Check(tmagic == magic, "invalid format,magic number mismatch");
    this->info.LoadBinary(fi);
    // load in the row data file
    fname += ".row.blob";
    utils::FileStream fs(utils::FopenCheck(fname.c_str(), "rb"));
    iter_->Load(fs);
    if (!silent) {
      utils::Printf("DMatrixPage: %lux%lu matrix is loaded",
                    static_cast<unsigned long>(info.num_row()),
                    static_cast<unsigned long>(info.num_col()));
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
  /*! \brief save a DataMatrix as DMatrixPage */
  inline static void Save(const char *fname_, const DataMatrix &mat, bool silent) {
    std::string fname = fname_;
    utils::FileStream fs(utils::FopenCheck(fname.c_str(), "wb"));
    int magic = kMagic;
    fs.Write(&magic, sizeof(magic));
    mat.info.SaveBinary(fs);
    fs.Close();
    fname += ".row.blob";
    utils::FileStream fbin(utils::FopenCheck(fname.c_str(), "wb"));
    ThreadRowPageIterator::Save(mat.fmat()->RowIterator(), fbin);
    fbin.Close();
    if (!silent) {
      utils::Printf("DMatrixPage: %lux%lu is saved to %s\n",
                    static_cast<unsigned long>(mat.info.num_row()),
                    static_cast<unsigned long>(mat.info.num_col()), fname_);
    }
  }
  /*! \brief save a LibSVM format file as DMatrixPage */
  inline void LoadText(const char *uri,
                       const char* cache_file,
                       bool silent,
                       bool loadsplit) {
    int rank = 0, npart = 1;
    if (loadsplit) {
      rank = rabit::GetRank();
      npart = rabit::GetWorldSize();
    }
    std::string fname_row = std::string(cache_file) + ".row.blob";
    utils::FileStream fo(utils::FopenCheck(fname_row.c_str(), "wb"));
    RowBatchPage page(ThreadRowPageIterator::kPageSize);
    dmlc::InputSplit *in =
        dmlc::InputSplit::Create(uri, rank, npart);
    std::string line;
    info.Clear();
    while (in->ReadRecord(&line)) {
      float label;
      std::istringstream ss(line);
      std::vector<RowBatch::Entry> feats;
      ss >> label;
      while (!ss.eof()) {
        RowBatch::Entry e;
        if (!(ss >> e.index)) break;
        ss.ignore(32, ':');
        if (!(ss >> e.fvalue)) break;
        feats.push_back(e);
      }
      RowBatch::Inst row(BeginPtr(feats), feats.size());
      if (!page.PushRow(row)) {
        page.Save(fo);
        page.Clear();
        utils::Check(page.PushRow(row), "row is too big");
      }
      for (size_t i = 0; i < feats.size(); ++i) {
        info.info.num_col = std::max(info.info.num_col,
                                     static_cast<size_t>(feats[i].index+1));
      }
      this->info.labels.push_back(label);
      info.info.num_row += 1;
    }
    if (page.Size() != 0) {
      page.Save(fo);
    }
    delete in;
    fo.Close();    
    iter_->Load(utils::FileStream(utils::FopenCheck(fname_row.c_str(), "rb")));    
    // save data matrix
    utils::FileStream fs(utils::FopenCheck(cache_file, "wb"));
    int magic = kMagic;
    fs.Write(&magic, sizeof(magic));
    this->info.SaveBinary(fs);
    fs.Close();
    if (!silent) {
      utils::Printf("DMatrixPage: %lux%lu is parsed from %s\n",
                    static_cast<unsigned long>(info.num_row()),
                    static_cast<unsigned long>(info.num_col()),
                    uri);
    }
  }
  /*! \brief magic number used to identify DMatrix */
  static const int kMagic = TKMagic;

 protected:
  /*! \brief row iterator */
  ThreadRowPageIterator *iter_;
};

class DMatrixPage : public DMatrixPageBase<0xffffab02> {
 public:
  DMatrixPage(void) {
    fmat_ = new FMatrixS(iter_);
  }
  virtual ~DMatrixPage(void) {
    delete fmat_;
  }
  virtual IFMatrix *fmat(void) const {
    return fmat_;
  }
  /*! \brief the real fmatrix */
  IFMatrix *fmat_;
};
}  // namespace io
}  // namespace xgboost
#endif  // XGBOOST_IO_PAGE_ROW_ITER_INL_HPP_
