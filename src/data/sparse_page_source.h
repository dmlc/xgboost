/*!
 *  Copyright (c) 2014-2021 by Contributors
 * \file sparse_page_source.h
 */
#ifndef XGBOOST_DATA_SPARSE_PAGE_SOURCE_H_
#define XGBOOST_DATA_SPARSE_PAGE_SOURCE_H_

#include <algorithm>  // std::min
#include <string>
#include <utility>
#include <vector>
#include <future>
#include <thread>
#include <map>
#include <memory>

#if (defined _WIN32) || (defined __CYGWIN__)
#include <locale>  // std::local
#include <cctype>  // std::isalpha
#endif  // (defined _WIN32) || (defined __CYGWIN__)

#include "rabit/rabit.h"
#include "xgboost/base.h"
#include "xgboost/data.h"

#include "adapter.h"
#include "sparse_page_writer.h"
#include "proxy_dmatrix.h"

#include "../common/common.h"


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

inline void TryDeleteCacheFile(const std::string& file) {
  if (std::remove(file.c_str()) != 0) {
    LOG(WARNING) << "Couldn't remove external memory cache file " << file
              << "; you may want to remove it manually";
  }
}

struct Cache {
  // whether the write to the cache is complete
  bool written;
  std::string name;
  std::string format;
  // offset into binary cache file.
  std::vector<size_t> offset;

  Cache(bool w, std::string n, std::string fmt)
      : written{w}, name{std::move(n)}, format{std::move(fmt)} {
    offset.push_back(0);
  }

  static std::string ShardName(std::string name, std::string format) {
    CHECK_EQ(format.front(), '.');
    return name + format;
  }

  std::string ShardName() {
    return ShardName(this->name, this->format);
  }

  // The write is completed.
  void Commit() {
    if (!written) {
      std::partial_sum(offset.begin(), offset.end(), offset.begin());
      written = true;
    }
  }
};

// Prevents multi-threaded call.
class TryLockGuard {
  std::mutex& lock_;

 public:
  explicit TryLockGuard(std::mutex& lock) : lock_{lock} {  // NOLINT
    CHECK(lock_.try_lock()) << "Multiple threads attempting to use Sparse DMatrix.";
  }
  ~TryLockGuard() {
    lock_.unlock();
  }
};

template <typename S>
class SparsePageSourceImpl : public BatchIteratorImpl<S> {
 protected:
  // Prevents calling this iterator from multiple places(or threads).
  std::mutex single_threaded_;

  std::shared_ptr<S> page_;

  bool at_end_ {false};
  float missing_;
  int nthreads_;
  bst_feature_t n_features_;

  uint32_t count_{0};

  size_t n_batches_ {0};

  std::shared_ptr<Cache> cache_info_;
  std::unique_ptr<dmlc::Stream> fo_;

  using Ring = std::vector<std::future<std::shared_ptr<S>>>;
  // A ring storing futures to data.  Since the DMatrix iterator is forward only, so we
  // can pre-fetch data in a ring.
  std::unique_ptr<Ring> ring_{new Ring};

  bool ReadCache() {
    CHECK(!at_end_);
    if (!cache_info_->written) {
      return false;
    }
    if (fo_) {
      fo_.reset();
      ring_->resize(n_batches_);
    }
    // An heuristic for number of pre-fetched batches.  We can make it part of BatchParam
    // to let user adjust number of pre-fetched batches when needed.
    size_t constexpr kPreFetch = 4;

    size_t n_prefetch_batches = std::min(kPreFetch, n_batches_);
    CHECK_GT(n_prefetch_batches, 0) << n_batches_;
    size_t fetch_it = count_;
    for (size_t i = 0; i < n_prefetch_batches; ++i, ++fetch_it) {
      fetch_it %= n_batches_;  // ring
      if (ring_->at(fetch_it).valid()) { continue; }
      auto const *self = this;  // make sure it's const
      CHECK_LT(fetch_it, cache_info_->offset.size());
      ring_->at(fetch_it) = std::async(std::launch::async, [fetch_it, self]() {
        std::unique_ptr<SparsePageFormat<S>> fmt{CreatePageFormat<S>("raw")};
        auto n = self->cache_info_->ShardName();
        size_t offset = self->cache_info_->offset.at(fetch_it);
        std::unique_ptr<dmlc::SeekStream> fi{
            dmlc::SeekStream::CreateForRead(n.c_str())};
        fi->Seek(offset);
        CHECK_EQ(fi->Tell(), offset);
        auto page = std::make_shared<S>();
        CHECK(fmt->Read(page.get(), fi.get()));
        return page;
      });
    }
    page_ = (*ring_)[count_].get();
    return true;
  }

  void WriteCache() {
    if (!cache_info_->written) {
      std::unique_ptr<SparsePageFormat<S>> fmt{CreatePageFormat<S>("raw")};
      if (!fo_) {
        auto n = cache_info_->ShardName();
        fo_.reset(dmlc::Stream::Create(n.c_str(), "w"));
      }
      auto bytes = fmt->Write(*page_, fo_.get());
      cache_info_->offset.push_back(bytes);
    }
  }

  virtual void Fetch() = 0;

 public:
  SparsePageSourceImpl(float missing, int nthreads, bst_feature_t n_features,
                       size_t n_batches, std::shared_ptr<Cache> cache)
      : missing_{missing}, nthreads_{nthreads}, n_features_{n_features},
        n_batches_{n_batches}, cache_info_{std::move(cache)} {}

  SparsePageSourceImpl(SparsePageSourceImpl const &that) = delete;

  ~SparsePageSourceImpl() override {
    for (auto& fu : *ring_) {
      if (fu.valid()) {
        fu.get();
      }
    }
  }

  uint32_t Iter() const { return count_; }

  S &operator*() override {
    CHECK(page_);
    return *page_;
  }
  const S &operator*() const override {
    CHECK(page_);
    return *page_;
  }
  std::shared_ptr<S> Page() const {
    return page_;
  }

  bool AtEnd() const override {
    return at_end_;
  }

  virtual void Reset() {
    TryLockGuard guard{single_threaded_};
    at_end_ = false;
    count_ = 0;
    this->Fetch();
  }
};

#if defined(XGBOOST_USE_CUDA)
void DevicePush(DMatrixProxy* proxy, float missing, SparsePage* page);
#else
inline void DevicePush(DMatrixProxy* proxy, float missing, SparsePage* page) {
  common::AssertGPUSupport();
}
#endif

class SparsePageSource : public SparsePageSourceImpl<SparsePage> {
  DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext> iter_;
  DMatrixProxy* proxy_;
  size_t base_row_id_ {0};

  void Fetch() final {
    page_.reset(new SparsePage{});
    if (!this->ReadCache()) {
      bool type_error { false };
      CHECK(proxy_);
      HostAdapterDispatch(proxy_, [&](auto const &value) {
        page_->Push(value, this->missing_, this->nthreads_);
      }, &type_error);
      if (type_error) {
        DevicePush(proxy_, missing_, page_.get());
      }
      page_->SetBaseRowId(base_row_id_);
      base_row_id_ += page_->Size();
      n_batches_++;
      this->WriteCache();
    }
  }

 public:
  SparsePageSource(
      DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext> iter,
      DMatrixProxy *proxy, float missing, int nthreads,
      bst_feature_t n_features, size_t n_batches, std::shared_ptr<Cache> cache)
      : SparsePageSourceImpl(missing, nthreads, n_features, n_batches, cache),
        iter_{iter}, proxy_{proxy} {
    if (!cache_info_->written) {
      iter_.Reset();
      iter_.Next();
    }
    this->Fetch();
  }

  void operator++() final {
    TryLockGuard guard{single_threaded_};
    count_++;
    if (cache_info_->written) {
      at_end_ = (count_ == n_batches_);
    } else {
      at_end_ = !iter_.Next();;
    }

    if (at_end_) {
      cache_info_->Commit();
      if (n_batches_ != 0) {
        CHECK_EQ(count_, n_batches_);
      }
      CHECK_GE(count_, 1);
      proxy_ = nullptr;
    } else {
      this->Fetch();
    }
  }

  void Reset() override {
    if (proxy_) {
      TryLockGuard guard{single_threaded_};
      iter_.Reset();
    }
    SparsePageSourceImpl::Reset();

    TryLockGuard guard{single_threaded_};
    base_row_id_ = 0;
  }
};

// A mixin for advancing the iterator.
template <typename S>
class PageSourceIncMixIn : public SparsePageSourceImpl<S> {
 protected:
  std::shared_ptr<SparsePageSource> source_;

 public:
  using SparsePageSourceImpl<S>::SparsePageSourceImpl;
  void operator++() final {
    TryLockGuard guard{this->single_threaded_};
    ++(*source_);

    ++this->count_;
    this->at_end_ = source_->AtEnd();

    if (this->at_end_) {
      this->cache_info_->Commit();
      if (this->n_batches_ != 0) {
        CHECK_EQ(this->count_, this->n_batches_);
      }
      CHECK_GE(this->count_, 1);
    } else {
      this->Fetch();
    }
    CHECK_EQ(source_->Iter(), this->count_);
  }
};

class CSCPageSource : public PageSourceIncMixIn<CSCPage> {
 protected:
  void Fetch() final {
    if (!this->ReadCache()) {
      auto const &csr = source_->Page();
      this->page_.reset(new CSCPage{});
      this->page_->PushCSC(csr->GetTranspose(n_features_));
      page_->SetBaseRowId(csr->base_rowid);
      this->WriteCache();
    }
  }

 public:
  CSCPageSource(
      float missing, int nthreads, bst_feature_t n_features, size_t n_batches,
      std::shared_ptr<Cache> cache,
      std::shared_ptr<SparsePageSource> source)
      : PageSourceIncMixIn(missing, nthreads, n_features,
                                     n_batches, cache) {
    this->source_ = source;
    this->Fetch();
  }
};

class SortedCSCPageSource : public PageSourceIncMixIn<SortedCSCPage> {
 protected:
  void Fetch() final {
    if (!this->ReadCache()) {
      auto const &csr = this->source_->Page();
      this->page_.reset(new SortedCSCPage{});
      this->page_->PushCSC(csr->GetTranspose(n_features_));
      CHECK_EQ(this->page_->Size(), n_features_);
      CHECK_EQ(this->page_->data.Size(), csr->data.Size());
      this->page_->SortRows();
      page_->SetBaseRowId(csr->base_rowid);
      this->WriteCache();
    }
  }

 public:
  SortedCSCPageSource(float missing, int nthreads, bst_feature_t n_features,
                      size_t n_batches, std::shared_ptr<Cache> cache,
                      std::shared_ptr<SparsePageSource> source)
      : PageSourceIncMixIn(missing, nthreads, n_features, n_batches, cache) {
    this->source_ = source;
    this->Fetch();
  }
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_PAGE_SOURCE_H_
