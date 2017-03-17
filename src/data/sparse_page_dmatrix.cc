/*!
 * Copyright 2014 by Contributors
 * \file sparse_page_dmatrix.cc
 * \brief The external memory version of Page Iterator.
 * \author Tianqi Chen
 */
#include <dmlc/base.h>
#include <dmlc/timer.h>
#include <xgboost/logging.h>
#include <memory>

#if DMLC_ENABLE_STD_THREAD
#include "./sparse_page_dmatrix.h"
#include "../common/random.h"
#include "../common/common.h"
#include "../common/group_data.h"

namespace xgboost {
namespace data {

SparsePageDMatrix::ColPageIter::ColPageIter(
    std::vector<std::unique_ptr<dmlc::SeekStream> >&& files)
    : page_(nullptr), clock_ptr_(0), files_(std::move(files)) {
  load_all_ = false;
  formats_.resize(files_.size());
  prefetchers_.resize(files_.size());

  for (size_t i = 0; i < files_.size(); ++i) {
    dmlc::SeekStream* fi = files_[i].get();
    std::string format;
    CHECK(fi->Read(&format)) << "Invalid page format";
    formats_[i].reset(SparsePage::Format::Create(format));
    SparsePage::Format* fmt = formats_[i].get();
    size_t fbegin = fi->Tell();
    prefetchers_[i].reset(new dmlc::ThreadedIter<SparsePage>(4));
    prefetchers_[i]->Init([this, fi, fmt] (SparsePage** dptr) {
        if (*dptr == nullptr) {
          *dptr = new SparsePage();
        }
        if (load_all_) {
          return fmt->Read(*dptr, fi);
        } else {
          return fmt->Read(*dptr, fi, index_set_);
        }
      },  [this, fi, fbegin] () {
        fi->Seek(fbegin);
        index_set_ = set_index_set_;
        load_all_ = set_load_all_;
      });
  }
}

SparsePageDMatrix::ColPageIter::~ColPageIter() {
  delete page_;
}

bool SparsePageDMatrix::ColPageIter::Next() {
  // doing clock rotation over shards.
  if (page_ != nullptr) {
    size_t n = prefetchers_.size();
    prefetchers_[(clock_ptr_ + n - 1) % n]->Recycle(&page_);
  }
  if (prefetchers_[clock_ptr_]->Next(&page_)) {
    out_.col_index = dmlc::BeginPtr(index_set_);
    col_data_.resize(page_->offset.size() - 1, SparseBatch::Inst(nullptr, 0));
    for (size_t i = 0; i < col_data_.size(); ++i) {
      col_data_[i] = SparseBatch::Inst
          (dmlc::BeginPtr(page_->data) + page_->offset[i],
           static_cast<bst_uint>(page_->offset[i + 1] - page_->offset[i]));
    }
    out_.col_data = dmlc::BeginPtr(col_data_);
    out_.size = col_data_.size();
    // advance clock
    clock_ptr_ = (clock_ptr_ + 1) % prefetchers_.size();
    return true;
  } else {
    return false;
  }
}

void SparsePageDMatrix::ColPageIter::BeforeFirst() {
  clock_ptr_ = 0;
  for (auto& p : prefetchers_) {
    p->BeforeFirst();
  }
}

void SparsePageDMatrix::ColPageIter::Init(const std::vector<bst_uint>& index_set,
                                          bool load_all) {
  set_index_set_ = index_set;
  set_load_all_ = load_all;
  std::sort(set_index_set_.begin(), set_index_set_.end());
  this->BeforeFirst();
}

dmlc::DataIter<ColBatch>* SparsePageDMatrix::ColIterator() {
  CHECK(col_iter_.get() != nullptr);
  std::vector<bst_uint> col_index;
  size_t ncol = this->info().num_col;
  for (size_t i = 0; i < ncol; ++i) {
    col_index.push_back(static_cast<bst_uint>(i));
  }
  col_iter_->Init(col_index, true);
  return col_iter_.get();
}

dmlc::DataIter<ColBatch>* SparsePageDMatrix::
ColIterator(const std::vector<bst_uint>& fset) {
  CHECK(col_iter_.get() != nullptr);
  std::vector<bst_uint> col_index;
  size_t ncol = this->info().num_col;
  for (size_t i = 0; i < fset.size(); ++i) {
    if (fset[i] < ncol) {
      col_index.push_back(fset[i]);
    }
  }
  col_iter_->Init(col_index, false);
  return col_iter_.get();
}


bool SparsePageDMatrix::TryInitColData() {
  // load meta data.
  std::vector<std::string> cache_shards = common::Split(cache_info_, ':');
  {
    std::string col_meta_name = cache_shards[0] + ".col.meta";
    std::unique_ptr<dmlc::Stream> fmeta(
        dmlc::Stream::Create(col_meta_name.c_str(), "r", true));
    if (fmeta.get() == nullptr) return false;
    CHECK(fmeta->Read(&buffered_rowset_)) << "invalid col.meta file";
    CHECK(fmeta->Read(&col_size_)) << "invalid col.meta file";
  }
  // load real data
  std::vector<std::unique_ptr<dmlc::SeekStream> > files;
  for (const std::string& prefix : cache_shards) {
    std::string col_data_name = prefix + ".col.page";
    std::unique_ptr<dmlc::SeekStream> fdata(
        dmlc::SeekStream::CreateForRead(col_data_name.c_str(), true));
    if (fdata.get() == nullptr) return false;
    files.push_back(std::move(fdata));
  }
  col_iter_.reset(new ColPageIter(std::move(files)));
  return true;
}

void SparsePageDMatrix::InitColAccess(const std::vector<bool>& enabled,
                                      float pkeep,
                                      size_t max_row_perbatch) {
  if (HaveColAccess()) return;
  if (TryInitColData()) return;

  const MetaInfo& info = this->info();
  if (max_row_perbatch == std::numeric_limits<size_t>::max()) {
    max_row_perbatch = kMaxRowPerBatch;
  }
  buffered_rowset_.clear();
  col_size_.resize(info.num_col);
  std::fill(col_size_.begin(), col_size_.end(), 0);
  dmlc::DataIter<RowBatch>* iter = this->RowIterator();
  std::bernoulli_distribution coin_flip(pkeep);
  size_t batch_ptr = 0, batch_top = 0;
  SparsePage tmp;
  auto& rnd = common::GlobalRandom();

  // function to create the page.
  auto make_col_batch = [&] (
      const SparsePage& prow,
      size_t begin,
      SparsePage *pcol) {
    pcol->Clear();
    pcol->min_index = buffered_rowset_[begin];
    const int nthread = std::max(omp_get_max_threads(), std::max(omp_get_num_procs() / 2 - 1, 1));
    common::ParallelGroupBuilder<SparseBatch::Entry>
    builder(&pcol->offset, &pcol->data);
    builder.InitBudget(info.num_col, nthread);
    bst_omp_uint ndata = static_cast<bst_uint>(prow.Size());
    #pragma omp parallel for schedule(static) num_threads(nthread)
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      int tid = omp_get_thread_num();
      for (size_t j = prow.offset[i]; j < prow.offset[i+1]; ++j) {
        const SparseBatch::Entry &e = prow.data[j];
        if (enabled[e.index]) {
          builder.AddBudget(e.index, tid);
        }
      }
    }
    builder.InitStorage();
    #pragma omp parallel for schedule(static) num_threads(nthread)
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      int tid = omp_get_thread_num();
      for (size_t j = prow.offset[i]; j < prow.offset[i+1]; ++j) {
        const SparseBatch::Entry &e = prow.data[j];
        builder.Push(e.index,
                     SparseBatch::Entry(buffered_rowset_[i + begin], e.fvalue),
                     tid);
      }
    }
    CHECK_EQ(pcol->Size(), info.num_col);
    // sort columns
    bst_omp_uint ncol = static_cast<bst_omp_uint>(pcol->Size());
    #pragma omp parallel for schedule(dynamic, 1) num_threads(nthread)
    for (bst_omp_uint i = 0; i < ncol; ++i) {
      if (pcol->offset[i] < pcol->offset[i + 1]) {
        std::sort(dmlc::BeginPtr(pcol->data) + pcol->offset[i],
                  dmlc::BeginPtr(pcol->data) + pcol->offset[i + 1],
                  SparseBatch::Entry::CmpValue);
      }
    }
  };

  auto make_next_col = [&] (SparsePage* dptr) {
    tmp.Clear();
    size_t btop = buffered_rowset_.size();

    while (true) {
      if (batch_ptr != batch_top) {
        const RowBatch& batch = iter->Value();
        CHECK_EQ(batch_top, batch.size);
        for (size_t i = batch_ptr; i < batch_top; ++i) {
          bst_uint ridx = static_cast<bst_uint>(batch.base_rowid + i);
          if (pkeep == 1.0f || coin_flip(rnd)) {
            buffered_rowset_.push_back(ridx);
            tmp.Push(batch[i]);
          }

          if (tmp.Size() >= max_row_perbatch ||
              tmp.MemCostBytes() >= kPageSize) {
            make_col_batch(tmp, btop, dptr);
            batch_ptr = i + 1;
            return true;
          }
        }
        batch_ptr = batch_top;
      }
      if (!iter->Next()) break;
      batch_ptr = 0;
      batch_top = iter->Value().size;
    }

    if (tmp.Size() != 0) {
      make_col_batch(tmp, btop, dptr);
      return true;
    } else {
      return false;
    }
  };

  std::vector<std::string> cache_shards = common::Split(cache_info_, ':');
  std::vector<std::string> name_shards, format_shards;
  for (const std::string& prefix : cache_shards) {
    name_shards.push_back(prefix + ".col.page");
    format_shards.push_back(SparsePage::Format::DecideFormat(prefix).second);
  }

  {
    SparsePage::Writer writer(name_shards, format_shards, 6);
    std::shared_ptr<SparsePage> page;
    writer.Alloc(&page); page->Clear();

    double tstart = dmlc::GetTime();
    size_t bytes_write = 0;
    // print every 4 sec.
    const double kStep = 4.0;
    size_t tick_expected = kStep;

    while (make_next_col(page.get())) {
      for (size_t i = 0; i < page->Size(); ++i) {
        col_size_[i] += page->offset[i + 1] - page->offset[i];
      }

      bytes_write += page->MemCostBytes();
      writer.PushWrite(std::move(page));
      writer.Alloc(&page);
      page->Clear();

      double tdiff = dmlc::GetTime() - tstart;
      if (tdiff >= tick_expected) {
        LOG(CONSOLE) << "Writing col.page file to " << cache_info_
                     << " in " << ((bytes_write >> 20UL) / tdiff) << " MB/s, "
                     << (bytes_write >> 20UL) << " MB writen";
        tick_expected += kStep;
      }
    }
    // save meta data
    std::string col_meta_name = cache_shards[0] + ".col.meta";
    std::unique_ptr<dmlc::Stream> fo(
        dmlc::Stream::Create(col_meta_name.c_str(), "w"));
    fo->Write(buffered_rowset_);
    fo->Write(col_size_);
    fo.reset(nullptr);
  }
  // initialize column data
  CHECK(TryInitColData());
}

}  // namespace data
}  // namespace xgboost
#endif
