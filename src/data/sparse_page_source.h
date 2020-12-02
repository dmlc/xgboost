/*!
 *  Copyright (c) 2014-2019 by Contributors
 * \file page_csr_source.h
 *  External memory data source, saved with sparse_batch_page binary format.
 * \author Tianqi Chen
 *
 * -------------------------------------------------
 * Random notes on implementation of external memory
 * -------------------------------------------------
 *
 * As of XGBoost 1.3, the general pipeline is:
 *
 * dmlc text file parser --> file adapter --> sparse page source -> data pool -->
 * write to binary cache --> load it back ~~> [ other pages (csc, ellpack, sorted csc) -->
 * write to binary cache ] --> use it in various algorithms.
 *
 *       ~~> means optional
 *
 * The dmlc text file parser returns number of blocks based on available threads, which
 * can make the data partitioning non-deterministic, so here we set up an extra data pool
 * to stage parsed data.  As a result, the number of blocks returned by text parser does
 * not equal to number of blocks in binary cache.
 *
 * Binary cache loading is async by the dmlc threaded iterator, which helps performance,
 * but as this iterator itself is not thread safe, so calling
 * `dmatrix->GetBatches<SparsePage>` is also not thread safe.  Please note that, the
 * threaded iterator is also used inside dmlc text file parser.
 *
 * Memory consumption is difficult to control due to various reasons.  Firstly the text
 * parsing doesn't have a batch size, only a hard coded buffer size is available.
 * Secondly, everything is loaded/written with async queue, with multiple queues running
 * the memory consumption is difficult to measure.
 *
 * The threaded iterator relies heavily on C++ memory model and threading primitive.  The
 * concurrent writer for binary cache is an old copy of moody queue.  We should try to
 * replace them with something more robust.
 */
#ifndef XGBOOST_DATA_SPARSE_PAGE_SOURCE_H_
#define XGBOOST_DATA_SPARSE_PAGE_SOURCE_H_

#include <dmlc/threadediter.h>
#include <dmlc/timer.h>

#include <algorithm>
#include <limits>
#include <locale>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

#include "rabit/rabit.h"
#include "xgboost/base.h"
#include "xgboost/data.h"

#include "adapter.h"
#include "sparse_page_writer.h"
#include "../common/common.h"
#include <xgboost/data.h>

namespace detail {

// Split a cache info string with delimiter ':'
// If cache info string contains drive letter (e.g. C:), exclude it before splitting
inline std::vector<std::string>
GetCacheShards(const std::string& cache_info) {
#if (defined _WIN32) || (defined __CYGWIN__)
  if (cache_info.length() >= 2
      && std::isalpha(cache_info[0], std::locale::classic())
      && cache_info[1] == ':') {
    std::vector<std::string> cache_shards
      = xgboost::common::Split(cache_info.substr(2), ':');
    cache_shards[0] = cache_info.substr(0, 2) + cache_shards[0];
    return cache_shards;
  }
#endif  // (defined _WIN32) || (defined __CYGWIN__)
  return xgboost::common::Split(cache_info, ':');
}

}  // namespace detail

namespace xgboost {
namespace data {

template<typename S, typename T>
class SparseBatchIteratorImpl : public BatchIteratorImpl<T> {
 public:
  explicit SparseBatchIteratorImpl(S* source) : source_(source) {
    CHECK(source_ != nullptr);
    source_->BeforeFirst();
    source_->Next();
  }
  T& operator*() override { return source_->Value(); }
  const T& operator*() const override { return source_->Value(); }
  void operator++() override { at_end_ = !source_->Next(); }
  bool AtEnd() const override { return at_end_; }

 private:
  S* source_{nullptr};
  bool at_end_{ false };
};

  /*! \brief magic number used to identify Page */
  static const int kMagic = 0xffffab02;
/*!
 * \brief decide the format from cache prefix.
 * \return pair of row format, column format type of the cache prefix.
 */
inline std::pair<std::string, std::string> DecideFormat(const std::string& cache_prefix) {
  size_t pos = cache_prefix.rfind(".fmt-");

  if (pos != std::string::npos) {
    std::string fmt = cache_prefix.substr(pos + 5, cache_prefix.length());
    size_t cpos = fmt.rfind('-');
    if (cpos != std::string::npos) {
      return std::make_pair(fmt.substr(0, cpos), fmt.substr(cpos + 1, fmt.length()));
    } else {
      return std::make_pair(fmt, fmt);
    }
  } else {
    std::string raw = "raw";
    return std::make_pair(raw, raw);
  }
}

struct CacheInfo {
  std::string name_info;
  std::vector<std::string> format_shards;
  std::vector<std::string> name_shards;
};

inline CacheInfo ParseCacheInfo(const std::string& cache_info, const std::string& page_type) {
  CacheInfo info;
  std::vector<std::string> cache_shards = ::detail::GetCacheShards(cache_info);
  CHECK_NE(cache_shards.size(), 0U);
  // read in the info files.
  info.name_info = cache_shards[0];
  for (const std::string& prefix : cache_shards) {
    info.name_shards.push_back(prefix + page_type);
    info.format_shards.push_back(DecideFormat(prefix).first);
  }
  return info;
}

inline void TryDeleteCacheFile(const std::string& file) {
  if (std::remove(file.c_str()) != 0) {
    LOG(WARNING) << "Couldn't remove external memory cache file " << file
              << "; you may want to remove it manually";
  }
}

inline void CheckCacheFileExists(const std::string& file) {
  std::ifstream f(file.c_str());
  if (f.good()) {
    LOG(FATAL)
        << "Cache file " << file << " exists already;  "
        << "Is there another DMatrix with the same "
           "cache prefix?  It can be caused by previously used DMatrix that "
           "hasn't been collected by language environment garbage collector.  "
           "Otherwise please remove it manually.";
  }
}

/**
 * \brief Given a set of cache files and page type, this object iterates over batches
 * using prefetching for improved performance. Not thread safe.
 *
 * \tparam  PageT Type of the page t.
 */
template <typename PageT>
class ExternalMemoryPrefetcher : dmlc::DataIter<PageT> {
 public:
  explicit ExternalMemoryPrefetcher(const CacheInfo& info) noexcept(false)
      : base_rowid_(0), page_(nullptr), clock_ptr_(0) {
    // read in the info files
    CHECK_NE(info.name_shards.size(), 0U);
    {
      std::unique_ptr<dmlc::Stream> finfo(
          dmlc::Stream::Create(info.name_info.c_str(), "r"));
      int tmagic;
      CHECK(finfo->Read(&tmagic));
      CHECK_EQ(tmagic, kMagic) << "invalid format, magic number mismatch";
    }
    files_.resize(info.name_shards.size());
    formats_.resize(info.name_shards.size());
    prefetchers_.resize(info.name_shards.size());

    // read in the cache files.
    for (size_t i = 0; i < info.name_shards.size(); ++i) {
      std::string name_row = info.name_shards.at(i);
      files_[i].reset(dmlc::SeekStream::CreateForRead(name_row.c_str()));
      std::unique_ptr<dmlc::SeekStream>& fi = files_[i];
      std::string format;
      CHECK(fi->Read(&format)) << "Invalid page format";
      formats_[i].reset(CreatePageFormat<PageT>(format));
      std::unique_ptr<SparsePageFormat<PageT>>& fmt = formats_[i];
      size_t fbegin = fi->Tell();
      prefetchers_[i].reset(new dmlc::ThreadedIter<PageT>(4));
      prefetchers_[i]->Init(
          [&fi, &fmt](PageT** dptr) {
            if (*dptr == nullptr) {
              *dptr = new PageT();
            }
            return fmt->Read(*dptr, fi.get());
          },
          [&fi, fbegin]() { fi->Seek(fbegin); });
    }
  }
  /*! \brief destructor */
  ~ExternalMemoryPrefetcher() override {
    delete page_;
  }

  // implement Next
  bool Next() override {
    CHECK(mutex_.try_lock()) << "Multiple threads attempting to use prefetcher";
    // doing clock rotation over shards.
    if (page_ != nullptr) {
      size_t n = prefetchers_.size();
      prefetchers_[(clock_ptr_ + n - 1) % n]->Recycle(&page_);
    }

    if (prefetchers_[clock_ptr_]->Next(&page_)) {
      page_->SetBaseRowId(base_rowid_);
      base_rowid_ += page_->Size();
      // advance clock
      clock_ptr_ = (clock_ptr_ + 1) % prefetchers_.size();
      mutex_.unlock();
      return true;
    } else {
      mutex_.unlock();
      return false;
    }
  }

  // implement BeforeFirst
  void BeforeFirst() override {
    CHECK(mutex_.try_lock()) << "Multiple threads attempting to use prefetcher";
    base_rowid_ = 0;
    clock_ptr_ = 0;
    for (auto& p : prefetchers_) {
      p->BeforeFirst();
    }
    mutex_.unlock();
  }

  // implement Value
  PageT& Value() { return *page_; }

  const PageT& Value() const override { return *page_; }

 private:
  std::mutex mutex_;
  /*! \brief number of rows */
  size_t base_rowid_;
  /*! \brief page currently on hold. */
  PageT* page_;
  /*! \brief internal clock ptr */
  size_t clock_ptr_;
  /*! \brief file pointer to the row blob file. */
  std::vector<std::unique_ptr<dmlc::SeekStream>> files_;
  /*! \brief Sparse page format file. */
  std::vector<std::unique_ptr<SparsePageFormat<PageT>>> formats_;
  /*! \brief internal prefetcher. */
  std::vector<std::unique_ptr<dmlc::ThreadedIter<PageT>>> prefetchers_;
};


// A data pool to keep the size of each page balanced and data partitioning to be
// deterministic.
class DataPool {
  size_t inferred_num_rows_;
  MetaInfo* info_;
  SparsePage pool_;
  size_t page_size_;
  SparsePageWriter<SparsePage> *writer_;

  void Slice(std::shared_ptr<SparsePage> out, size_t offset, size_t n_rows,
             size_t entry_offset) const;
  void SplitWritePage();

 public:
  DataPool(MetaInfo *info, size_t page_size,
           SparsePageWriter<SparsePage> *writer)
      : inferred_num_rows_{0}, info_{info},
        page_size_{page_size}, writer_{writer} {}

  void Push(std::shared_ptr<SparsePage> page) {
    info_->num_nonzero_ += page->data.Size();
    pool_.Push(*page);
    if (pool_.Size() > page_size_) {
      this->SplitWritePage();
    }
    page->Clear();
  }

  size_t Finalize();
};

class SparsePageSource {
 public:
  template <typename AdapterT>
  SparsePageSource(AdapterT* adapter, float missing, int nthread,
                   const std::string& cache_info,
                   const size_t page_size = DMatrix::kPageSize) {
    const std::string page_type = ".row.page";
    cache_info_ = ParseCacheInfo(cache_info, page_type);

    // Warn user if old cache files
    CheckCacheFileExists(cache_info_.name_info);
    for (auto file : cache_info_.name_shards) {
      CheckCacheFileExists(file);
    }

    {
      SparsePageWriter<SparsePage> writer(cache_info_.name_shards,
                                          cache_info_.format_shards, 6);
      DataPool pool(&info, page_size, &writer);

      std::shared_ptr<SparsePage> page { new SparsePage };

      uint64_t inferred_num_columns = 0;
      uint64_t inferred_num_rows = 0;

      const uint64_t default_max = std::numeric_limits<uint64_t>::max();
      uint64_t last_group_id = default_max;
      bst_uint group_size = 0;
      std::vector<uint64_t> qids;
      adapter->BeforeFirst();
      while (adapter->Next()) {
        auto& batch = adapter->Value();
        if (batch.Labels() != nullptr) {
          auto& labels = info.labels_.HostVector();
          labels.insert(labels.end(), batch.Labels(),
                        batch.Labels() + batch.Size());
        }
        if (batch.Weights() != nullptr) {
          auto& weights = info.weights_.HostVector();
          weights.insert(weights.end(), batch.Weights(),
                         batch.Weights() + batch.Size());
        }
        if (batch.BaseMargin() != nullptr) {
          auto& base_margin = info.base_margin_.HostVector();
          base_margin.insert(base_margin.end(), batch.BaseMargin(),
                             batch.BaseMargin() + batch.Size());
        }
        if (batch.Qid() != nullptr) {
          qids.insert(qids.end(), batch.Qid(), batch.Qid() + batch.Size());
          // get group
          for (size_t i = 0; i < batch.Size(); ++i) {
            const uint64_t cur_group_id = batch.Qid()[i];
            if (last_group_id == default_max ||
                last_group_id != cur_group_id) {
              info.group_ptr_.push_back(group_size);
            }
            last_group_id = cur_group_id;
            ++group_size;
          }
        }
        CHECK_EQ(page->Size(), 0);
        auto batch_max_columns = page->Push(batch, missing, nthread);
        inferred_num_columns =
            std::max(batch_max_columns, inferred_num_columns);
        inferred_num_rows += page->Size();
        pool.Push(page);
        page->SetBaseRowId(inferred_num_rows);
      }

      if (last_group_id != default_max) {
        if (group_size > info.group_ptr_.back()) {
          info.group_ptr_.push_back(group_size);
        }
      }

      // Deal with empty rows/columns if necessary
      if (adapter->NumColumns() == kAdapterUnknownSize) {
        info.num_col_ = inferred_num_columns;
      } else {
        info.num_col_ = adapter->NumColumns();
      }
      // Synchronise worker columns
      rabit::Allreduce<rabit::op::Max>(&info.num_col_, 1);

      if (adapter->NumRows() == kAdapterUnknownSize) {
        info.num_row_ = inferred_num_rows;
      } else {
        if (page->offset.HostVector().empty()) {
          page->offset.HostVector().emplace_back(0);
        }

        while (inferred_num_rows < adapter->NumRows()) {
          page->offset.HostVector().emplace_back(
              page->offset.HostVector().back());
          inferred_num_rows++;
        }
        info.num_row_ = adapter->NumRows();
      }

      pool.Push(page);
      pool.Finalize();

      std::unique_ptr<dmlc::Stream> fo(
          dmlc::Stream::Create(cache_info_.name_info.c_str(), "w"));
      int tmagic = kMagic;
      fo->Write(tmagic);
      // Either every row has query ID or none at all
      CHECK(qids.empty() || qids.size() == info.num_row_);
      info.SaveBinary(fo.get());
    }
    LOG(INFO) << "SparsePageSource Finished writing to "
              << cache_info_.name_info;

    external_prefetcher_.reset(
        new ExternalMemoryPrefetcher<SparsePage>(cache_info_));
  }

  ~SparsePageSource() {
    external_prefetcher_.reset();
    TryDeleteCacheFile(cache_info_.name_info);
    for (auto file : cache_info_.name_shards) {
      TryDeleteCacheFile(file);
    }
  }

  BatchSet<SparsePage> GetBatchSet() {
    auto begin_iter = BatchIterator<SparsePage>(
        new SparseBatchIteratorImpl<ExternalMemoryPrefetcher<SparsePage>,
                                    SparsePage>(external_prefetcher_.get()));
    return BatchSet<SparsePage>(begin_iter);
  }
  MetaInfo info;

 private:
  std::unique_ptr<ExternalMemoryPrefetcher<SparsePage>> external_prefetcher_;
  CacheInfo cache_info_;
};

class CSCPageSource {
 public:
  CSCPageSource(DMatrix* src, const std::string& cache_info,
                const size_t page_size = DMatrix::kPageSize) {
    std::string page_type = ".col.page";
    cache_info_ = ParseCacheInfo(cache_info, page_type);
    for (auto file : cache_info_.name_shards) {
      CheckCacheFileExists(file);
    }
    {
      SparsePageWriter<SparsePage> writer(cache_info_.name_shards,
                                          cache_info_.format_shards, 6);
      std::shared_ptr<SparsePage> page;
      writer.Alloc(&page);
      page->Clear();

      size_t bytes_write = 0;
      double tstart = dmlc::GetTime();
      for (auto& batch : src->GetBatches<SparsePage>()) {
        page->PushCSC(batch.GetTranspose(src->Info().num_col_));

        if (page->MemCostBytes() >= page_size) {
          bytes_write += page->MemCostBytes();
          writer.PushWrite(std::move(page));
          writer.Alloc(&page);
          page->Clear();
          double tdiff = dmlc::GetTime() - tstart;
          LOG(INFO) << "Writing to " << cache_info << " in "
                    << ((bytes_write >> 20UL) / tdiff) << " MB/s, "
                    << (bytes_write >> 20UL) << " written";
        }
      }
      if (page->data.Size() != 0) {
        writer.PushWrite(std::move(page));
      }
      LOG(INFO) << "CSCPageSource: Finished writing to "
                << cache_info_.name_info;
    }
    external_prefetcher_.reset(
        new ExternalMemoryPrefetcher<CSCPage>(cache_info_));
  }

  ~CSCPageSource() {
    external_prefetcher_.reset();
    for (auto file : cache_info_.name_shards) {
      TryDeleteCacheFile(file);
    }
  }

  BatchSet<CSCPage> GetBatchSet() {
    auto begin_iter = BatchIterator<CSCPage>(
        new SparseBatchIteratorImpl<ExternalMemoryPrefetcher<CSCPage>, CSCPage>(
            external_prefetcher_.get()));
    return BatchSet<CSCPage>(begin_iter);
  }

 private:
  std::unique_ptr<ExternalMemoryPrefetcher<CSCPage>> external_prefetcher_;
  CacheInfo cache_info_;
};

class SortedCSCPageSource {
 public:
  SortedCSCPageSource(DMatrix* src, const std::string& cache_info,
                      const size_t page_size = DMatrix::kPageSize) {
    std::string page_type = ".sorted.col.page";
    cache_info_ = ParseCacheInfo(cache_info, page_type);
    for (auto file : cache_info_.name_shards) {
      CheckCacheFileExists(file);
    }
    {
      SparsePageWriter<SparsePage> writer(cache_info_.name_shards,
                                          cache_info_.format_shards, 6);
      std::shared_ptr<SparsePage> page;
      writer.Alloc(&page);
      page->Clear();

      size_t bytes_write = 0;
      double tstart = dmlc::GetTime();
      for (auto& batch : src->GetBatches<SparsePage>()) {
        SparsePage tmp = batch.GetTranspose(src->Info().num_col_);
        page->PushCSC(tmp);
        page->SortRows();

        if (page->MemCostBytes() >= page_size) {
          bytes_write += page->MemCostBytes();
          writer.PushWrite(std::move(page));
          writer.Alloc(&page);
          page->Clear();
          double tdiff = dmlc::GetTime() - tstart;
          LOG(INFO) << "Writing to " << cache_info << " in "
                    << ((bytes_write >> 20UL) / tdiff) << " MB/s, "
                    << (bytes_write >> 20UL) << " written";
        }
      }
      if (page->data.Size() != 0) {
        writer.PushWrite(std::move(page));
      }
      LOG(INFO) << "SortedCSCPageSource: Finished writing to "
                << cache_info_.name_info;
    }
    external_prefetcher_.reset(
        new ExternalMemoryPrefetcher<SortedCSCPage>(cache_info_));
  }
  ~SortedCSCPageSource() {
    external_prefetcher_.reset();
    for (auto file : cache_info_.name_shards) {
      TryDeleteCacheFile(file);
    }
  }

  BatchSet<SortedCSCPage> GetBatchSet() {
    auto begin_iter = BatchIterator<SortedCSCPage>(
        new SparseBatchIteratorImpl<ExternalMemoryPrefetcher<SortedCSCPage>,
                                    SortedCSCPage>(external_prefetcher_.get()));
    return BatchSet<SortedCSCPage>(begin_iter);
  }

 private:
  std::unique_ptr<ExternalMemoryPrefetcher<SortedCSCPage>> external_prefetcher_;
  CacheInfo cache_info_;
};

}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_PAGE_SOURCE_H_
