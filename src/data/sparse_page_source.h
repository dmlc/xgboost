/*!
 *  Copyright (c) 2014 by Contributors
 * \file page_csr_source.h
 *  External memory data source, saved with sparse_batch_page binary format.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SPARSE_PAGE_SOURCE_H_
#define XGBOOST_DATA_SPARSE_PAGE_SOURCE_H_

#include <xgboost/base.h>
#include <xgboost/data.h>
#include <dmlc/threadediter.h>
#include <dmlc/timer.h>

#include <algorithm>
#include <limits>
#include <locale>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sparse_page_writer.h"

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
      formats_[i].reset(SparsePageFormat::Create(format));
      std::unique_ptr<SparsePageFormat>& fmt = formats_[i];
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
      page_->base_rowid = base_rowid_;
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

  /*!
   * \brief Create source by taking data from parser.
   * \param src source parser.
   * \param cache_info The cache_info of cache file location.
   * \param page_size Page size for external memory.
   */
  static void CreateRowPage(dmlc::Parser<uint32_t>* src,
                            const std::string& cache_info,
                            const size_t page_size = DMatrix::kPageSize) {
    const std::string page_type = ".row.page";
    std::vector<std::string> cache_shards = GetCacheShards(cache_info);
    CHECK_NE(cache_shards.size(), 0U);
    // read in the info files.
    std::string name_info = cache_shards[0];
    std::vector<std::string> name_shards, format_shards;
    for (const std::string& prefix : cache_shards) {
      name_shards.push_back(prefix + page_type);
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
          auto& labels = info.labels_.HostVector();
          labels.insert(labels.end(), batch.label, batch.label + batch.size);
        }
        if (batch.weight != nullptr) {
          auto& weights = info.weights_.HostVector();
          weights.insert(weights.end(), batch.weight, batch.weight + batch.size);
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
        if (page->MemCostBytes() >= page_size) {
          bytes_write += page->MemCostBytes();
          writer.PushWrite(std::move(page));
          writer.Alloc(&page);
          page->Clear();

          double tdiff = dmlc::GetTime() - tstart;
          if (tdiff >= tick_expected) {
            LOG(CONSOLE) << "Writing " << page_type << " to " << cache_info
                         << " in " << ((bytes_write >> 20UL) / tdiff) << " MB/s, "
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

      if (page->data.Size() != 0) {
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
    LOG(INFO) << "SparsePageSource::CreateRowPage Finished writing to "
              << name_info;
  }

  /*!
   * \brief Create source cache by copy content from DMatrix.
   * \param cache_info The cache_info of cache file location.
   */
  static void CreateRowPage(DMatrix* src,
                            const std::string& cache_info) {
    const std::string page_type = ".row.page";
    CreatePageFromDMatrix(src, cache_info, page_type);
  }

  /*!
   * \brief Create source cache by copy content from DMatrix. Creates transposed column page, may be sorted or not.
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
    std::vector<std::string> cache_shards = GetCacheShards(cache_info);
    CHECK_NE(cache_shards.size(), 0U);
    // read in the info files.
    std::string name_info = cache_shards[0];
    std::vector<std::string> name_shards, format_shards;
    for (const std::string& prefix : cache_shards) {
      name_shards.push_back(prefix + page_type);
      format_shards.push_back(SparsePageFormat::DecideFormat(prefix).first);
    }
    {
      SparsePageWriter writer(name_shards, format_shards, 6);
      std::shared_ptr<SparsePage> page;
      writer.Alloc(&page);
      page->Clear();

      MetaInfo info = src->Info();
      size_t bytes_write = 0;
      double tstart = dmlc::GetTime();
      for (auto& batch : src->GetBatches<SparsePage>()) {
        if (page_type == ".row.page") {
          page->Push(batch);
        } else if (page_type == ".col.page") {
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

      std::unique_ptr<dmlc::Stream> fo(
          dmlc::Stream::Create(name_info.c_str(), "w"));
      int tmagic = kMagic;
      fo->Write(&tmagic, sizeof(tmagic));
      info.SaveBinary(fo.get());
    }
    LOG(INFO) << "SparsePageSource: Finished writing to " << name_info;
  }

  /*! \brief number of rows */
  size_t base_rowid_;
  /*! \brief page currently on hold. */
  T *page_;
  /*! \brief internal clock ptr */
  size_t clock_ptr_;
  /*! \brief file pointer to the row blob file. */
  std::vector<std::unique_ptr<dmlc::SeekStream> > files_;
  /*! \brief Sparse page format file. */
  std::vector<std::unique_ptr<SparsePageFormat> > formats_;
  /*! \brief internal prefetcher. */
  std::vector<std::unique_ptr<dmlc::ThreadedIter<T> > > prefetchers_;
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_PAGE_SOURCE_H_
