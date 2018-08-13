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
    formats_[i].reset(SparsePageFormat::Create(format));
    SparsePageFormat* fmt = formats_[i].get();
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

void SparsePageDMatrix::ColPageIter::Init(
    const std::vector<bst_uint>& index_set) {
  set_index_set_ = index_set;
  set_load_all_ = true;
  std::sort(set_index_set_.begin(), set_index_set_.end());
  this->BeforeFirst();
}

  dmlc::DataIter<SparsePage>* SparsePageDMatrix::ColIterator() {
  CHECK(col_iter_ != nullptr);
  std::vector<bst_uint> col_index;
  std::iota(col_index.begin(), col_index.end(), bst_uint(0));
  col_iter_->Init(col_index);
  return col_iter_.get();
}

bool SparsePageDMatrix::TryInitColData(bool sorted) {
  // load meta data.
  std::vector<std::string> cache_shards = common::Split(cache_info_, ':');
  {
    std::string col_meta_name = cache_shards[0] + ".col.meta";
    std::unique_ptr<dmlc::Stream> fmeta(
        dmlc::Stream::Create(col_meta_name.c_str(), "r", true));
    if (fmeta == nullptr) return false;
    CHECK(fmeta->Read(&buffered_rowset_)) << "invalid col.meta file";
    CHECK(fmeta->Read(&col_size_)) << "invalid col.meta file";
  }
  // load real data
  std::vector<std::unique_ptr<dmlc::SeekStream> > files;
  for (const std::string& prefix : cache_shards) {
    std::string col_data_name = prefix + ".col.page";
    std::unique_ptr<dmlc::SeekStream> fdata(
        dmlc::SeekStream::CreateForRead(col_data_name.c_str(), true));
    if (fdata == nullptr) return false;
    files.push_back(std::move(fdata));
  }
  col_iter_.reset(new ColPageIter(std::move(files)));
  // warning: no attempt to check here whether the cached data was sorted
  col_iter_->sorted = sorted;
  return true;
}

void SparsePageDMatrix::InitColAccess(
  size_t max_row_perbatch, bool sorted) {
  if (HaveColAccess(sorted)) return;
  if (TryInitColData(sorted)) return;
  const MetaInfo& info = this->Info();
  if (max_row_perbatch == std::numeric_limits<size_t>::max()) {
    max_row_perbatch = kMaxRowPerBatch;
  }
  buffered_rowset_.Clear();
  col_size_.resize(info.num_col_);
  std::fill(col_size_.begin(), col_size_.end(), 0);
  auto iter = this->RowIterator();
  size_t batch_ptr = 0, batch_top = 0;
  SparsePage tmp;

  // function to create the page.
  auto make_col_batch = [&] (
      const SparsePage& prow,
      size_t begin,
      SparsePage *pcol) {
    pcol->Clear();
    pcol->base_rowid = buffered_rowset_[begin];
    const int nthread = std::max(omp_get_max_threads(), std::max(omp_get_num_procs() / 2 - 1, 1));
    common::ParallelGroupBuilder<Entry>
    builder(&pcol->offset, &pcol->data);
    builder.InitBudget(info.num_col_, nthread);
    bst_omp_uint ndata = static_cast<bst_uint>(prow.Size());
    #pragma omp parallel for schedule(static) num_threads(nthread)
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      int tid = omp_get_thread_num();
      for (size_t j = prow.offset[i]; j < prow.offset[i+1]; ++j) {
        const  auto e = prow.data[j];
        builder.AddBudget(e.index, tid);
      }
    }
    builder.InitStorage();
    #pragma omp parallel for schedule(static) num_threads(nthread)
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      int tid = omp_get_thread_num();
      for (size_t j = prow.offset[i]; j < prow.offset[i+1]; ++j) {
        const Entry &e = prow.data[j];
        builder.Push(e.index,
                     Entry(buffered_rowset_[i + begin], e.fvalue),
                     tid);
      }
    }
    CHECK_EQ(pcol->Size(), info.num_col_);
    // sort columns
    if (sorted) {
      auto ncol = static_cast<bst_omp_uint>(pcol->Size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthread)
      for (bst_omp_uint i = 0; i < ncol; ++i) {
        if (pcol->offset[i] < pcol->offset[i + 1]) {
          std::sort(dmlc::BeginPtr(pcol->data) + pcol->offset[i],
            dmlc::BeginPtr(pcol->data) + pcol->offset[i + 1],
            Entry::CmpValue);
        }
      }
    }
  };

  auto make_next_col = [&] (SparsePage* dptr) {
    tmp.Clear();
    size_t btop = buffered_rowset_.Size();

    while (true) {
      if (batch_ptr != batch_top) {
         auto &batch = iter->Value();
        CHECK_EQ(batch_top, batch.Size());
        for (size_t i = batch_ptr; i < batch_top; ++i) {
          auto ridx = static_cast<bst_uint>(batch.base_rowid + i);
          buffered_rowset_.PushBack(ridx);
          tmp.Push(batch[i]);

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
      batch_top = iter->Value().Size();
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
    format_shards.push_back(SparsePageFormat::DecideFormat(prefix).second);
  }

  {
    SparsePageWriter writer(name_shards, format_shards, 6);
    std::shared_ptr<SparsePage> page;
    writer.Alloc(&page); page->Clear();

    double tstart = dmlc::GetTime();
    size_t bytes_write = 0;
    // print every 4 sec.
    constexpr double kStep = 4.0;
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
  CHECK(TryInitColData(sorted));
}

}  // namespace data
}  // namespace xgboost
#endif
