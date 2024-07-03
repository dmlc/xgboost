/**
 *  Copyright 2014-2024, XGBoost Contributors
 * \file sparse_page_source.h
 */
#ifndef XGBOOST_DATA_SPARSE_PAGE_SOURCE_H_
#define XGBOOST_DATA_SPARSE_PAGE_SOURCE_H_

#include <algorithm>  // for min
#include <atomic>     // for atomic
#include <cstdint>    // for uint64_t
#include <future>     // for future
#include <memory>     // for unique_ptr
#include <mutex>      // for mutex
#include <string>     // for string
#include <utility>    // for pair, move
#include <vector>     // for vector

#if !defined(XGBOOST_USE_CUDA)
#include "../common/common.h"  // for AssertGPUSupport
#endif                         // !defined(XGBOOST_USE_CUDA)

#include "../common/io.h"           // for PrivateMmapConstStream
#include "../common/threadpool.h"   // for ThreadPool
#include "../common/timer.h"        // for Monitor, Timer
#include "proxy_dmatrix.h"          // for DMatrixProxy
#include "sparse_page_writer.h"     // for SparsePageFormat
#include "xgboost/base.h"           // for bst_feature_t
#include "xgboost/data.h"           // for SparsePage, CSCPage, SortedCSCPage
#include "xgboost/global_config.h"  // for GlobalConfigThreadLocalStore
#include "xgboost/logging.h"        // for CHECK_EQ

namespace xgboost::data {
void TryDeleteCacheFile(const std::string& file);

/**
 * @brief Information about the cache including path and page offsets.
 */
struct Cache {
  // whether the write to the cache is complete
  bool written;
  bool on_host;
  std::string name;
  std::string format;
  // offset into binary cache file.
  std::vector<bst_idx_t> offset;

  Cache(bool w, std::string n, std::string fmt, bool on_host)
      : written{w}, on_host{on_host}, name{std::move(n)}, format{std::move(fmt)} {
    offset.push_back(0);
  }

  [[nodiscard]] static std::string ShardName(std::string name, std::string format) {
    CHECK_EQ(format.front(), '.');
    return name + format;
  }

  [[nodiscard]] std::string ShardName() const {
    return ShardName(this->name, this->format);
  }
  [[nodiscard]] bool OnHost() const { return on_host; }
  /**
   * @brief Record a page with size of n_bytes.
   */
  void Push(bst_idx_t n_bytes) { offset.push_back(n_bytes); }
  /**
   * @brief Returns the view start and length for the i^th page.
   */
  [[nodiscard]] auto View(std::size_t i) const {
    std::uint64_t off = offset.at(i);
    std::uint64_t len = this->Bytes(i);
    return std::pair{off, len};
  }
  /**
   * @brief Get the number of bytes for the i^th page.
   */
  [[nodiscard]] bst_idx_t Bytes(std::size_t i) const { return offset.at(i + 1) - offset[i]; }
  /**
   * @brief Call this once the write for the cache is complete.
   */
  void Commit();
};

// Prevents multi-threaded call to `GetBatches`.
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

// Similar to `dmlc::OMPException`, but doesn't need the threads to be joined before rethrow
class ExceHandler {
  std::mutex mutex_;
  std::atomic<bool> flag_{false};
  std::exception_ptr curr_exce_{nullptr};

 public:
  template <typename Fn>
  decltype(auto) Run(Fn&& fn) noexcept(true) {
    try {
      return fn();
    } catch (dmlc::Error const& e) {
      std::lock_guard<std::mutex> guard{mutex_};
      if (!curr_exce_) {
        curr_exce_ = std::current_exception();
      }
      flag_ = true;
    } catch (std::exception const& e) {
      std::lock_guard<std::mutex> guard{mutex_};
      if (!curr_exce_) {
        curr_exce_ = std::current_exception();
      }
      flag_ = true;
    } catch (...) {
      std::lock_guard<std::mutex> guard{mutex_};
      if (!curr_exce_) {
        curr_exce_ = std::current_exception();
      }
      flag_ = true;
    }
    return std::invoke_result_t<Fn>();
  }

  void Rethrow() noexcept(false) {
    if (flag_) {
      CHECK(curr_exce_);
      std::rethrow_exception(curr_exce_);
    }
  }
};

/**
 * @brief Default implementation of the stream creater.
 */
template <typename S, template <typename> typename F>
class DefaultFormatStreamPolicy : public F<S> {
 public:
  using WriterT = common::AlignedFileWriteStream;
  using ReaderT = common::AlignedResourceReadStream;

 public:
  std::unique_ptr<WriterT> CreateWriter(StringView name, std::uint32_t iter) {
    std::unique_ptr<common::AlignedFileWriteStream> fo;
    if (iter == 0) {
      fo = std::make_unique<common::AlignedFileWriteStream>(name, "wb");
    } else {
      fo = std::make_unique<common::AlignedFileWriteStream>(name, "ab");
    }
    return fo;
  }

  std::unique_ptr<ReaderT> CreateReader(StringView name, std::uint64_t offset,
                                        std::uint64_t length) const {
    return std::make_unique<common::PrivateMmapConstStream>(std::string{name}, offset, length);
  }
};

/**
 * @brief Default implementatioin of the format creator.
 */
template <typename S>
class DefaultFormatPolicy {
 public:
  using FormatT = SparsePageFormat<S>;

 public:
  auto CreatePageFormat() const {
    std::unique_ptr<FormatT> fmt{::xgboost::data::CreatePageFormat<S>("raw")};
    return fmt;
  }
};

/**
 * @brief Base class for all page sources. Handles fetching, writing, and iteration.
 *
 * The interface to external storage is divided into two types. The first one is the
 * format, representing how to read and write the binary. The second part is where to
 * store the binary cache. These policies are implemented in the `FormatStreamPolicy`
 * policy class. The format policy controls how to create the format (the first part), and
 * the stream policy decides where the stream should read from and write to (the second
 * part). This way we can compose the polices and page types with ease.
 */
template <typename S,
          typename FormatStreamPolicy = DefaultFormatStreamPolicy<S, DefaultFormatPolicy>>
class SparsePageSourceImpl : public BatchIteratorImpl<S>, public FormatStreamPolicy {
 protected:
  // Prevents calling this iterator from multiple places(or threads).
  std::mutex single_threaded_;
  // The current page.
  std::shared_ptr<S> page_;
  // Workers for fetching data from external memory.
  common::ThreadPool workers_;

  bool at_end_ {false};
  float missing_;
  std::int32_t nthreads_;
  bst_feature_t n_features_;
  // Index to the current page.
  std::uint32_t count_{0};
  // Total number of batches.
  bst_idx_t n_batches_{0};

  std::shared_ptr<Cache> cache_info_;

  using Ring = std::vector<std::future<std::shared_ptr<S>>>;
  // A ring storing futures to data.  Since the DMatrix iterator is forward only, we can
  // pre-fetch data in a ring.
  std::unique_ptr<Ring> ring_{new Ring};
  // Catching exception in pre-fetch threads to prevent segfault. Not always work though,
  // OOM error can be delayed due to lazy commit. On the bright side, if mmap is used then
  // OOM error should be rare.
  ExceHandler exce_;
  common::Monitor monitor_;

  [[nodiscard]] bool ReadCache() {
    if (!cache_info_->written) {
      return false;
    }
    if (ring_->empty()) {
      ring_->resize(n_batches_);
    }
    // An heuristic for number of pre-fetched batches.  We can make it part of BatchParam
    // to let user adjust number of pre-fetched batches when needed.
    std::int32_t kPrefetches = 3;
    std::int32_t n_prefetches = std::min(nthreads_, kPrefetches);
    n_prefetches = std::max(n_prefetches, 1);
    std::int32_t n_prefetch_batches = std::min(static_cast<bst_idx_t>(n_prefetches), n_batches_);
    CHECK_GT(n_prefetch_batches, 0) << "total batches:" << n_batches_;
    CHECK_LE(n_prefetch_batches, kPrefetches);
    std::size_t fetch_it = count_;

    exce_.Rethrow();

    for (std::int32_t i = 0; i < n_prefetch_batches; ++i, ++fetch_it) {
      fetch_it %= n_batches_;  // ring
      if (ring_->at(fetch_it).valid()) {
        continue;
      }
      auto const* self = this;  // make sure it's const
      CHECK_LT(fetch_it, cache_info_->offset.size());
      ring_->at(fetch_it) = this->workers_.Submit([fetch_it, self, this] {
        auto page = std::make_shared<S>();
        this->exce_.Run([&] {
          std::unique_ptr<typename FormatStreamPolicy::FormatT> fmt{this->CreatePageFormat()};
          auto name = self->cache_info_->ShardName();
          auto [offset, length] = self->cache_info_->View(fetch_it);
          std::unique_ptr<typename FormatStreamPolicy::ReaderT> fi{
              this->CreateReader(name, offset, length)};
          CHECK(fmt->Read(page.get(), fi.get()));
        });
        return page;
      });
    }

    CHECK_EQ(std::count_if(ring_->cbegin(), ring_->cend(), [](auto const& f) { return f.valid(); }),
             n_prefetch_batches)
        << "Sparse DMatrix assumes forward iteration.";

    monitor_.Start("Wait");
    CHECK((*ring_)[count_].valid());
    page_ = (*ring_)[count_].get();
    CHECK(!(*ring_)[count_].valid());
    monitor_.Stop("Wait");

    exce_.Rethrow();

    return true;
  }

  void WriteCache() {
    CHECK(!cache_info_->written);
    common::Timer timer;
    timer.Start();
    auto fmt{this->CreatePageFormat()};

    auto name = cache_info_->ShardName();
    std::unique_ptr<typename FormatStreamPolicy::WriterT> fo{
        this->CreateWriter(StringView{name}, this->Iter())};
    auto bytes = fmt->Write(*page_, fo.get());

    timer.Stop();
    // Not entirely accurate, the kernels doesn't have to flush the data.
    LOG(INFO) << static_cast<double>(bytes) / 1024.0 / 1024.0 << " MB written in "
              << timer.ElapsedSeconds() << " seconds.";
    cache_info_->Push(bytes);
  }

  virtual void Fetch() = 0;

 public:
  SparsePageSourceImpl(float missing, int nthreads, bst_feature_t n_features, bst_idx_t n_batches,
                       std::shared_ptr<Cache> cache)
      : workers_{std::max(2, std::min(nthreads, 16)),
                 [config = *GlobalConfigThreadLocalStore::Get()] {
                   *GlobalConfigThreadLocalStore::Get() = config;
                 }},
        missing_{missing},
        nthreads_{nthreads},
        n_features_{n_features},
        n_batches_{n_batches},
        cache_info_{std::move(cache)} {
    monitor_.Init(typeid(S).name());  // not pretty, but works for basic profiling
  }

  SparsePageSourceImpl(SparsePageSourceImpl const &that) = delete;

  ~SparsePageSourceImpl() override {
    // Don't orphan the threads.
    for (auto& fu : *ring_) {
      if (fu.valid()) {
        fu.get();
      }
    }
  }

  [[nodiscard]] std::uint32_t Iter() const { return count_; }

  [[nodiscard]] S const& operator*() const override {
    CHECK(page_);
    return *page_;
  }

  [[nodiscard]] std::shared_ptr<S const> Page() const override {
    return page_;
  }

  [[nodiscard]] bool AtEnd() const override {
    return at_end_;
  }

  // Call this at the last iteration.
  void EndIter() {
    CHECK_EQ(this->cache_info_->offset.size(), this->n_batches_ + 1);
    this->cache_info_->Commit();
    if (this->n_batches_ != 0) {
      CHECK_EQ(this->count_, this->n_batches_);
    }
    CHECK_GE(this->count_, 1);
    this->count_ = 0;
  }

  virtual void Reset() {
    TryLockGuard guard{single_threaded_};

    this->at_end_ = false;
    auto cnt = this->count_;
    this->count_ = 0;
    if (cnt != 0) {
      // The last iteration did not get to the end, clear the ring to start from 0.
      this->ring_ = std::make_unique<Ring>();
      this->Fetch();
    }
  }
};

#if defined(XGBOOST_USE_CUDA)
// Push data from CUDA.
void DevicePush(DMatrixProxy* proxy, float missing, SparsePage* page);
#else
inline void DevicePush(DMatrixProxy*, float, SparsePage*) { common::AssertGPUSupport(); }
#endif

class SparsePageSource : public SparsePageSourceImpl<SparsePage> {
  // This is the source iterator from the user.
  DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext> iter_;
  DMatrixProxy* proxy_;
  std::size_t base_row_id_{0};
  bst_idx_t fetch_cnt_{0};  // Used for sanity check.

  void Fetch() final {
    fetch_cnt_++;
    page_ = std::make_shared<SparsePage>();
    // The first round of reading, this is responsible for initialization.
    if (!this->ReadCache()) {
      bool type_error{false};
      CHECK(proxy_);
      HostAdapterDispatch(
          proxy_,
          [&](auto const& adapter_batch) {
            page_->Push(adapter_batch, this->missing_, this->nthreads_);
          },
          &type_error);
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
      bst_feature_t n_features, uint32_t n_batches, std::shared_ptr<Cache> cache)
      : SparsePageSourceImpl(missing, nthreads, n_features, n_batches, cache),
        iter_{iter}, proxy_{proxy} {
    if (!cache_info_->written) {
      iter_.Reset();
      CHECK(iter_.Next()) << "Must have at least 1 batch.";
    }
    this->Fetch();
  }

  SparsePageSource& operator++() final {
    TryLockGuard guard{single_threaded_};
    count_++;

    if (cache_info_->written) {
      at_end_ = (count_ == n_batches_);
    } else {
      at_end_ = !iter_.Next();
    }
    CHECK_LE(count_, n_batches_);

    if (at_end_) {
      this->EndIter();
      this->proxy_ = nullptr;
    }

    this->Fetch();
    return *this;
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

  [[nodiscard]] auto FetchCount() const { return fetch_cnt_; }
};

// A mixin for advancing the iterator.
template <typename S,
          typename FormatCreatePolicy = DefaultFormatStreamPolicy<S, DefaultFormatPolicy>>
class PageSourceIncMixIn : public SparsePageSourceImpl<S, FormatCreatePolicy> {
 protected:
  std::shared_ptr<SparsePageSource> source_;
  using Super = SparsePageSourceImpl<S, FormatCreatePolicy>;
  // synchronize the row page, `hist` and `gpu_hist` don't need the original sparse page
  // so we avoid fetching it.
  bool sync_{true};

 public:
  PageSourceIncMixIn(float missing, std::int32_t nthreads, bst_feature_t n_features,
                     bst_idx_t n_batches, std::shared_ptr<Cache> cache, bool sync)
      : Super::SparsePageSourceImpl{missing, nthreads, n_features, n_batches, cache}, sync_{sync} {}
  // This function always operate on the source first, then the downstream. The downstream
  // can assume the source to be ready.
  [[nodiscard]] PageSourceIncMixIn& operator++() final {
    TryLockGuard guard{this->single_threaded_};
    // Increment the source.
    if (sync_) {
      ++(*source_);
    }
    // Increment self.
    ++this->count_;
    // Set at end.
    this->at_end_ = this->count_ == this->n_batches_;

    if (this->at_end_) {
      // If this is the first round of iterations, we have just built the binary cache
      // from soruce. For a non-sync page type, the source hasn't been updated to the end
      // iteration yet due to skipped increment. We increment the source here and it will
      // call the `EndIter` method itself.
      bool src_need_inc = !sync_ && this->source_->Iter() != 0;
      if (src_need_inc) {
        CHECK_EQ(this->source_->Iter(), this->count_ - 1);
        ++(*source_);
      }
      this->EndIter();

      if (src_need_inc) {
        CHECK(this->cache_info_->written);
      }
    }
    this->Fetch();

    if (sync_) {
      // Sanity check.
      CHECK_EQ(source_->Iter(), this->count_);
    }
    return *this;
  }

  void Reset() final {
    this->source_->Reset();
    Super::Reset();
  }
};

class CSCPageSource : public PageSourceIncMixIn<CSCPage> {
 protected:
  void Fetch() final {
    if (!this->ReadCache()) {
      auto const &csr = source_->Page();
      this->page_.reset(new CSCPage{});
      // we might be able to optimize this by merging transpose and pushcsc
      this->page_->PushCSC(csr->GetTranspose(n_features_, nthreads_));
      page_->SetBaseRowId(csr->base_rowid);
      this->WriteCache();
    }
  }

 public:
  CSCPageSource(float missing, int nthreads, bst_feature_t n_features, uint32_t n_batches,
                std::shared_ptr<Cache> cache, std::shared_ptr<SparsePageSource> source)
      : PageSourceIncMixIn(missing, nthreads, n_features, n_batches, cache, true) {
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
      // we might be able to optimize this by merging transpose and pushcsc
      this->page_->PushCSC(csr->GetTranspose(n_features_, nthreads_));
      CHECK_EQ(this->page_->Size(), n_features_);
      CHECK_EQ(this->page_->data.Size(), csr->data.Size());
      this->page_->SortRows(this->nthreads_);
      page_->SetBaseRowId(csr->base_rowid);
      this->WriteCache();
    }
  }

 public:
  SortedCSCPageSource(float missing, int nthreads, bst_feature_t n_features,
                      uint32_t n_batches, std::shared_ptr<Cache> cache,
                      std::shared_ptr<SparsePageSource> source)
      : PageSourceIncMixIn(missing, nthreads, n_features, n_batches, cache, true) {
    this->source_ = source;
    this->Fetch();
  }
};
}  // namespace xgboost::data
#endif  // XGBOOST_DATA_SPARSE_PAGE_SOURCE_H_
