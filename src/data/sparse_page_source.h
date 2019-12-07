/*!
 *  Copyright (c) 2014-2019 by Contributors
 * \file page_csr_source.h
 *  External memory data source, saved with sparse_batch_page binary format.
 * \author Tianqi Chen
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

#include "xgboost/base.h"
#include "xgboost/data.h"

#include "adapter.h"
#include "sparse_page_writer.h"
#include "../common/common.h"

namespace {

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

}  // anonymous namespace

namespace xgboost {
namespace data {

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
  std::vector<std::string> cache_shards = GetCacheShards(cache_info);
  CHECK_NE(cache_shards.size(), 0U);
  // read in the info files.
  info.name_info = cache_shards[0];
  for (const std::string& prefix : cache_shards) {
    info.name_shards.push_back(prefix + page_type);
    info.format_shards.push_back(DecideFormat(prefix).first);
  }
  return info;
}

/*!
 * \brief External memory data source.
 * \code
 * std::unique_ptr<DataSource> source(new SimpleCSRSource(cache_prefix));
 * // add data to source
 * DMatrix* dmat = DMatrix::Create(std::move(source));
 * \encode
 */
template<typename T>
class SparsePageSource : public DataSource<T> {
 public:
  /*!
   * \brief Create source from cache files the cache_prefix.
   * \param cache_prefix The prefix of cache we want to solve.
   */
  explicit SparsePageSource(const std::string& cache_info,
                            const std::string& page_type) noexcept(false)
      : base_rowid_(0), page_(nullptr), clock_ptr_(0) {
    // read in the info files
    std::vector<std::string> cache_shards = GetCacheShards(cache_info);
    CHECK_NE(cache_shards.size(), 0U);
    {
      std::string name_info = cache_shards[0];
      std::unique_ptr<dmlc::Stream> finfo(dmlc::Stream::Create(name_info.c_str(), "r"));
      int tmagic;
      CHECK_EQ(finfo->Read(&tmagic, sizeof(tmagic)), sizeof(tmagic));
      CHECK_EQ(tmagic, kMagic) << "invalid format, magic number mismatch";
      this->info.LoadBinary(finfo.get());
    }
    files_.resize(cache_shards.size());
    formats_.resize(cache_shards.size());
    prefetchers_.resize(cache_shards.size());

    // read in the cache files.
    for (size_t i = 0; i < cache_shards.size(); ++i) {
      std::string name_row = cache_shards[i] + page_type;
      files_[i].reset(dmlc::SeekStream::CreateForRead(name_row.c_str()));
      std::unique_ptr<dmlc::SeekStream>& fi = files_[i];
      std::string format;
      CHECK(fi->Read(&format)) << "Invalid page format";
      formats_[i].reset(CreatePageFormat<T>(format));
      std::unique_ptr<SparsePageFormat<T>>& fmt = formats_[i];
      size_t fbegin = fi->Tell();
      prefetchers_[i].reset(new dmlc::ThreadedIter<T>(4));
      prefetchers_[i]->Init([&fi, &fmt] (T** dptr) {
        if (*dptr == nullptr) {
          *dptr = new T();
        }
        return fmt->Read(*dptr, fi.get());
      }, [&fi, fbegin] () { fi->Seek(fbegin); });
    }
  }

  /*! \brief destructor */
  ~SparsePageSource() override {
    delete page_;
  }

  // implement Next
  bool Next() override {
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
      return true;
    } else {
      return false;
    }
  }

  // implement BeforeFirst
  void BeforeFirst() override {
    base_rowid_ = 0;
    clock_ptr_ = 0;
    for (auto& p : prefetchers_) {
      p->BeforeFirst();
    }
  }

  // implement Value
  T& Value() {
    return *page_;
  }

  const T& Value() const override {
    return *page_;
  }

  template <typename AdapterT>
  static void CreateRowPage(AdapterT* adapter, float missing, int nthread,
                            const std::string& cache_info,
                            const size_t page_size = DMatrix::kPageSize) {
    const std::string page_type = ".row.page";
    auto cinfo = ParseCacheInfo(cache_info, page_type);
    {
      SparsePageWriter<SparsePage> writer(cinfo.name_shards,
                                          cinfo.format_shards, 6);
      std::shared_ptr<SparsePage> page;
      writer.Alloc(&page);
      page->Clear();

      uint64_t inferred_num_columns = 0;
      uint64_t inferred_num_rows = 0;
      MetaInfo info;
      size_t bytes_write = 0;
      double tstart = dmlc::GetTime();
      // print every 4 sec.
      constexpr double kStep = 4.0;
      size_t tick_expected = static_cast<double>(kStep);

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
        if (batch.Qid() != nullptr) {
          qids.insert(qids.end(), batch.Qid(), batch.Qid() + batch.Size());
          // get group
          for (size_t i = 0; i < batch.Size(); ++i) {
            const uint64_t cur_group_id = batch.Qid()[i];
            if (last_group_id == default_max || last_group_id != cur_group_id) {
              info.group_ptr_.push_back(group_size);
            }
            last_group_id = cur_group_id;
            ++group_size;
          }
        }
        auto batch_max_columns = page->Push(batch, missing, nthread);
        inferred_num_columns =
            std::max(batch_max_columns, inferred_num_columns);
        if (page->MemCostBytes() >= page_size) {
          inferred_num_rows += page->Size();
          info.num_nonzero_ += page->offset.HostVector().back();
          bytes_write += page->MemCostBytes();
          writer.PushWrite(std::move(page));
          writer.Alloc(&page);
          page->Clear();
          page->SetBaseRowId(inferred_num_rows);

          double tdiff = dmlc::GetTime() - tstart;
          if (tdiff >= tick_expected) {
            LOG(CONSOLE) << "Writing " << page_type << " to " << cache_info
                         << " in " << ((bytes_write >> 20UL) / tdiff)
                         << " MB/s, " << (bytes_write >> 20UL) << " written";
            tick_expected += static_cast<size_t>(kStep);
          }
        }
      }

      if (last_group_id != default_max) {
        if (group_size > info.group_ptr_.back()) {
          info.group_ptr_.push_back(group_size);
        }
      }
      inferred_num_rows += page->Size();
      if (!page->offset.HostVector().empty()) {
        info.num_nonzero_ += page->offset.HostVector().back();
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

      // Make sure we have at least one page if the dataset is empty
      if (page->data.Size() > 0 || info.num_row_ == 0) {
        writer.PushWrite(std::move(page));
      }
      std::unique_ptr<dmlc::Stream> fo(
          dmlc::Stream::Create(cinfo.name_info.c_str(), "w"));
      int tmagic = kMagic;
      fo->Write(&tmagic, sizeof(tmagic));
      // Either every row has query ID or none at all
      CHECK(qids.empty() || qids.size() == info.num_row_);
      info.SaveBinary(fo.get());
    }
    LOG(INFO) << "SparsePageSource::CreateRowPage Finished writing to "
              << cinfo.name_info;
  }
  /*!
   * \brief Create source cache by copy content from DMatrix.
   * Creates transposed column page, may be sorted or not.
   * \param cache_info The cache_info of cache file location.
   * \param sorted Whether columns should be pre-sorted
   */
  static void CreateColumnPage(DMatrix* src,
                               const std::string& cache_info, bool sorted) {
    const std::string page_type = sorted ? ".sorted.col.page" : ".col.page";
    CreatePageFromDMatrix(src, cache_info, page_type);
  }

  /*!
   * \brief Check if the cache file already exists.
   * \param cache_info The cache prefix of files.
   * \param page_type   Type of the page.
   * \return Whether cache file already exists.
   */
  static bool CacheExist(const std::string& cache_info,
                         const std::string& page_type) {
    std::vector<std::string> cache_shards = GetCacheShards(cache_info);
    CHECK_NE(cache_shards.size(), 0U);
    {
      std::string name_info = cache_shards[0];
      std::unique_ptr<dmlc::Stream> finfo(dmlc::Stream::Create(name_info.c_str(), "r", true));
      if (finfo == nullptr) return false;
    }
    for (const std::string& prefix : cache_shards) {
      std::string name_row = prefix + page_type;
      std::unique_ptr<dmlc::Stream> frow(dmlc::Stream::Create(name_row.c_str(), "r", true));
      if (frow == nullptr) return false;
    }
    return true;
  }

  /*! \brief magic number used to identify Page */
  static const int kMagic = 0xffffab02;

 private:
  static void CreatePageFromDMatrix(DMatrix* src, const std::string& cache_info,
                                    const std::string& page_type,
                                    const size_t page_size = DMatrix::kPageSize) {
    auto cinfo = ParseCacheInfo(cache_info, page_type);
    {
      SparsePageWriter<SparsePage> writer(cinfo.name_shards, cinfo.format_shards, 6);
      std::shared_ptr<SparsePage> page;
      writer.Alloc(&page);
      page->Clear();

      MetaInfo info = src->Info();
      size_t bytes_write = 0;
      double tstart = dmlc::GetTime();
      for (auto& batch : src->GetBatches<SparsePage>()) {
        if (page_type == ".col.page") {
          page->PushCSC(batch.GetTranspose(src->Info().num_col_));
        } else if (page_type == ".sorted.col.page") {
          SparsePage tmp = batch.GetTranspose(src->Info().num_col_);
          page->PushCSC(tmp);
          page->SortRows();
        } else {
          LOG(FATAL) << "Unknown page type: " << page_type;
        }

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
    }
    LOG(INFO) << "SparsePageSource: Finished writing to " << cinfo.name_info;
  }

  /*! \brief number of rows */
  size_t base_rowid_;
  /*! \brief page currently on hold. */
  T* page_;
  /*! \brief internal clock ptr */
  size_t clock_ptr_;
  /*! \brief file pointer to the row blob file. */
  std::vector<std::unique_ptr<dmlc::SeekStream>> files_;
  /*! \brief Sparse page format file. */
  std::vector<std::unique_ptr<SparsePageFormat<T>>> formats_;
  /*! \brief internal prefetcher. */
  std::vector<std::unique_ptr<dmlc::ThreadedIter<T>>> prefetchers_;
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_PAGE_SOURCE_H_
