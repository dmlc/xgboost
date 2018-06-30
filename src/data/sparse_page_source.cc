/*!
 * Copyright 2015 by Contributors
 * \file sparse_page_source.cc
 */
#include <dmlc/base.h>
#include <dmlc/timer.h>
#include <xgboost/logging.h>
#include <memory>

#if DMLC_ENABLE_STD_THREAD
#include "./sparse_page_source.h"
#include "../common/common.h"

namespace xgboost {
namespace data {

SparsePageSource::SparsePageSource(const std::string& cache_info)
    : base_rowid_(0), page_(nullptr), clock_ptr_(0) {
  // read in the info files
  std::vector<std::string> cache_shards = common::Split(cache_info, ':');
  CHECK_NE(cache_shards.size(), 0U);
  {
    std::string name_info = cache_shards[0];
    std::unique_ptr<dmlc::Stream> finfo(dmlc::Stream::Create(name_info.c_str(), "r"));
    int tmagic;
    CHECK_EQ(finfo->Read(&tmagic, sizeof(tmagic)), sizeof(tmagic));
    this->info.LoadBinary(finfo.get());
  }
  files_.resize(cache_shards.size());
  formats_.resize(cache_shards.size());
  prefetchers_.resize(cache_shards.size());

  // read in the cache files.
  for (size_t i = 0; i < cache_shards.size(); ++i) {
    std::string name_row = cache_shards[i] + ".row.page";
    files_[i].reset(dmlc::SeekStream::CreateForRead(name_row.c_str()));
    dmlc::SeekStream* fi = files_[i].get();
    std::string format;
    CHECK(fi->Read(&format)) << "Invalid page format";
    formats_[i].reset(SparsePageFormat::Create(format));
    SparsePageFormat* fmt = formats_[i].get();
    size_t fbegin = fi->Tell();
    prefetchers_[i].reset(new dmlc::ThreadedIter<SparsePage>(4));
    prefetchers_[i]->Init([fi, fmt] (SparsePage** dptr) {
        if (*dptr == nullptr) {
          *dptr = new SparsePage();
        }
        return fmt->Read(*dptr, fi);
      }, [fi, fbegin] () { fi->Seek(fbegin); });
  }
}

SparsePageSource::~SparsePageSource() {
  delete page_;
}

bool SparsePageSource::Next() {
  // doing clock rotation over shards.
  if (page_ != nullptr) {
    size_t n = prefetchers_.size();
    prefetchers_[(clock_ptr_ + n - 1) % n]->Recycle(&page_);
  }
  if (prefetchers_[clock_ptr_]->Next(&page_)) {
    page_->base_rowid = base_rowid_;
    base_rowid_ += page_->Size();
    // advance clock
    clock_ptr_ = (clock_ptr_ + 1) % prefetchers_.size();
    return true;
  } else {
    return false;
  }
}

void SparsePageSource::BeforeFirst() {
  base_rowid_ = 0;
  clock_ptr_ = 0;
  for (auto& p : prefetchers_) {
    p->BeforeFirst();
  }
}

const SparsePage& SparsePageSource::Value() const {
  return *page_;
}

bool SparsePageSource::CacheExist(const std::string& cache_info) {
  std::vector<std::string> cache_shards = common::Split(cache_info, ':');
  CHECK_NE(cache_shards.size(), 0U);
  {
    std::string name_info = cache_shards[0];
    std::unique_ptr<dmlc::Stream> finfo(dmlc::Stream::Create(name_info.c_str(), "r", true));
    if (finfo == nullptr) return false;
  }
  for (const std::string& prefix : cache_shards) {
    std::string name_row = prefix + ".row.page";
    std::unique_ptr<dmlc::Stream> frow(dmlc::Stream::Create(name_row.c_str(), "r", true));
    if (frow == nullptr) return false;
  }
  return true;
}

void SparsePageSource::Create(dmlc::Parser<uint32_t>* src,
                              const std::string& cache_info) {
  std::vector<std::string> cache_shards = common::Split(cache_info, ':');
  CHECK_NE(cache_shards.size(), 0U);
  // read in the info files.
  std::string name_info = cache_shards[0];
  std::vector<std::string> name_shards, format_shards;
  for (const std::string& prefix : cache_shards) {
    name_shards.push_back(prefix + ".row.page");
    format_shards.push_back(SparsePageFormat::DecideFormat(prefix).first);
  }
  {
    SparsePageWriter writer(name_shards, format_shards, 6);
    std::shared_ptr<SparsePage> page;
    writer.Alloc(&page); page->Clear();

    MetaInfo info;
    size_t bytes_write = 0;
    double tstart = dmlc::GetTime();
    // print every 4 sec.
    constexpr double kStep = 4.0;
    size_t tick_expected = static_cast<double>(kStep);

    const uint64_t default_max = std::numeric_limits<uint64_t>::max();
    uint64_t last_group_id = default_max;
    bst_uint group_size = 0;

    while (src->Next()) {
      const dmlc::RowBlock<uint32_t>& batch = src->Value();
      if (batch.label != nullptr) {
        info.labels_.insert(info.labels_.end(), batch.label, batch.label + batch.size);
      }
      if (batch.weight != nullptr) {
        info.weights_.insert(info.weights_.end(), batch.weight, batch.weight + batch.size);
      }
      if (batch.qid != nullptr) {
        info.qids_.insert(info.qids_.end(), batch.qid, batch.qid + batch.size);
        // get group
        for (size_t i = 0; i < batch.size; ++i) {
          const uint64_t cur_group_id = batch.qid[i];
          if (last_group_id == default_max || last_group_id != cur_group_id) {
            info.group_ptr_.push_back(group_size);
          }
          last_group_id = cur_group_id;
          ++group_size;
        }
      }
      info.num_row_ += batch.size;
      info.num_nonzero_ +=  batch.offset[batch.size] - batch.offset[0];
      for (size_t i = batch.offset[0]; i < batch.offset[batch.size]; ++i) {
        uint32_t index = batch.index[i];
        info.num_col_ = std::max(info.num_col_,
                                static_cast<uint64_t>(index + 1));
      }
      page->Push(batch);
      if (page->MemCostBytes() >= kPageSize) {
        bytes_write += page->MemCostBytes();
        writer.PushWrite(std::move(page));
        writer.Alloc(&page);
        page->Clear();

        double tdiff = dmlc::GetTime() - tstart;
        if (tdiff >= tick_expected) {
          LOG(CONSOLE) << "Writing row.page to " << cache_info << " in "
                       << ((bytes_write >> 20UL) / tdiff) << " MB/s, "
                       << (bytes_write >> 20UL) << " written";
          tick_expected += static_cast<size_t>(kStep);
        }
      }
    }
    if (last_group_id != default_max) {
      if (group_size > info.group_ptr_.back()) {
        info.group_ptr_.push_back(group_size);
      }
    }

    if (page->data.size() != 0) {
      writer.PushWrite(std::move(page));
    }

    std::unique_ptr<dmlc::Stream> fo(
        dmlc::Stream::Create(name_info.c_str(), "w"));
    int tmagic = kMagic;
    fo->Write(&tmagic, sizeof(tmagic));
    // Either every row has query ID or none at all
    CHECK(info.qids_.empty() || info.qids_.size() == info.num_row_);
    info.SaveBinary(fo.get());
  }
  LOG(CONSOLE) << "SparsePageSource: Finished writing to " << name_info;
}

void SparsePageSource::Create(DMatrix* src,
                              const std::string& cache_info) {
  std::vector<std::string> cache_shards = common::Split(cache_info, ':');
  CHECK_NE(cache_shards.size(), 0U);
  // read in the info files.
  std::string name_info = cache_shards[0];
  std::vector<std::string> name_shards, format_shards;
  for (const std::string& prefix : cache_shards) {
    name_shards.push_back(prefix + ".row.page");
    format_shards.push_back(SparsePageFormat::DecideFormat(prefix).first);
  }
  {
    SparsePageWriter writer(name_shards, format_shards, 6);
    std::shared_ptr<SparsePage> page;
    writer.Alloc(&page); page->Clear();

    MetaInfo info = src->Info();
    size_t bytes_write = 0;
    double tstart = dmlc::GetTime();
    auto iter = src->RowIterator();

    while (iter->Next()) {
      page->Push(iter->Value());
      if (page->MemCostBytes() >= kPageSize) {
        bytes_write += page->MemCostBytes();
        writer.PushWrite(std::move(page));
        writer.Alloc(&page);
        page->Clear();
        double tdiff = dmlc::GetTime() - tstart;
        LOG(CONSOLE) << "Writing to " << cache_info << " in "
                     << ((bytes_write >> 20UL) / tdiff) << " MB/s, "
                     << (bytes_write >> 20UL) << " written";
      }
    }
    if (page->data.size() != 0) {
      writer.PushWrite(std::move(page));
    }

    std::unique_ptr<dmlc::Stream> fo(
        dmlc::Stream::Create(name_info.c_str(), "w"));
    int tmagic = kMagic;
    fo->Write(&tmagic, sizeof(tmagic));
    info.SaveBinary(fo.get());
  }
  LOG(CONSOLE) << "SparsePageSource: Finished writing to " << name_info;
}

}  // namespace data
}  // namespace xgboost
#endif
