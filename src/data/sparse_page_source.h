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
#include <limits>     // for numeric_limits
#include <map>        // for map
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
#include "xgboost/global_config.h"  // for InitNewThread
#include "xgboost/logging.h"        // for CHECK_EQ

namespace xgboost::data {
void TryDeleteCacheFile(const std::string& file);

std::string MakeCachePrefix(std::string cache_prefix);

auto constexpr InvalidPageSize() { return std::numeric_limits<bst_idx_t>::max(); }

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
      : written{w}, on_host{on_host}, name{std::move(n)}, format{std::move(fmt)}, offset{0} {}

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
  /**
   * @brief Returns the number of pages in the cache.
   */
  [[nodiscard]] bst_idx_t Size() const { return this->offset.size() - 1; }
};

inline void DeleteCacheFiles(std::map<std::string, std::shared_ptr<Cache>> const& cache_info) {
  for (auto const& kv : cache_info) {
    CHECK(kv.second);
    auto n = kv.second->ShardName();
    if (kv.second->OnHost()) {
      continue;
    }
    TryDeleteCacheFile(n);
  }
}

[[nodiscard]] inline std::string MakeId(std::string prefix, void const* ptr) {
  std::stringstream ss;
  ss << ptr;
  return prefix + "-" + ss.str();
}

/**
 * @brief Make cache if it doesn't exist yet.
 */
[[nodiscard]] inline std::string MakeCache(void const* ptr, std::string format, bool on_host,
                                           std::string prefix,
                                           std::map<std::string, std::shared_ptr<Cache>>* out) {
  auto& cache_info = *out;
  auto name = MakeId(std::move(prefix), ptr);
  auto id = name + format;
  auto it = cache_info.find(id);
  if (it == cache_info.cend()) {
    cache_info[id].reset(new Cache{false, name, format, on_host});
    if (!on_host) {
      LOG(INFO) << "Make cache:" << cache_info[id]->ShardName();
    }
  }
  return id;
}

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
  auto CreatePageFormat(BatchParam const&) const {
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
  bst_idx_t fetch_cnt_{0};  // Used for sanity check.
  // Index to the current page.
  std::uint32_t count_{0};
  // How we pre-fetch the data.
  BatchParam param_;

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
    auto n_batches = this->cache_info_->Size();
    if (ring_->empty()) {
      ring_->resize(n_batches);
    }

    std::int32_t n_prefetches = std::min(nthreads_, this->param_.n_prefetch_batches);
    n_prefetches = std::max(n_prefetches, 1);
    std::int32_t n_prefetch_batches = std::min(static_cast<bst_idx_t>(n_prefetches), n_batches);
    CHECK_GT(n_prefetch_batches, 0);
    CHECK_LE(n_prefetch_batches, this->param_.n_prefetch_batches);
    std::size_t fetch_it = this->count_;

    exce_.Rethrow();
    // Clear out the existing page before loading new ones. This helps reduce memory usage
    // when page is not loaded with mmap, in addition, it triggers necessary CUDA
    // synchronizations by freeing memory.
    page_.reset();

    for (std::int32_t i = 0; i < n_prefetch_batches; ++i, ++fetch_it) {
      bool restart = fetch_it == n_batches;
      fetch_it %= n_batches;  // ring
      if (ring_->at(fetch_it).valid()) {
        continue;
      }
      auto const* self = this;  // make sure it's const
      CHECK_LT(fetch_it, cache_info_->offset.size());
      // Make sure the new iteration starts with a copy to avoid spilling configuration.
      if (restart) {
        this->param_.prefetch_copy = true;
      }
      ring_->at(fetch_it) = this->workers_.Submit([fetch_it, self, this] {
        auto page = std::make_shared<S>();
        this->exce_.Run([&] {
          std::unique_ptr<typename FormatStreamPolicy::FormatT> fmt{
              self->CreatePageFormat(self->param_)};
          auto name = self->cache_info_->ShardName();
          auto [offset, length] = self->cache_info_->View(fetch_it);
          std::unique_ptr<typename FormatStreamPolicy::ReaderT> fi{
              self->CreateReader(name, offset, length)};
          CHECK(fmt->Read(page.get(), fi.get()));
        });
        return page;
      });
      this->fetch_cnt_++;
    }

    CHECK_EQ(std::count_if(ring_->cbegin(), ring_->cend(), [](auto const& f) { return f.valid(); }),
             n_prefetch_batches)
        << "Sparse DMatrix assumes forward iteration.";

    monitor_.Start("Wait-" + std::to_string(count_));
    CHECK((*ring_)[count_].valid());
    page_ = (*ring_)[count_].get();
    monitor_.Stop("Wait-" + std::to_string(count_));

    exce_.Rethrow();

    return true;
  }

  void WriteCache() {
    CHECK(!cache_info_->written);
    common::Timer timer;
    timer.Start();
    auto fmt{this->CreatePageFormat(this->param_)};

    auto name = cache_info_->ShardName();
    std::unique_ptr<typename FormatStreamPolicy::WriterT> fo{
        this->CreateWriter(StringView{name}, this->Iter())};
    auto bytes = fmt->Write(*page_, fo.get());

    timer.Stop();
    if (bytes != InvalidPageSize()) {
      // Not entirely accurate, the kernels doesn't have to flush the data.
      LOG(INFO) << common::HumanMemUnit(bytes) << " written in " << timer.ElapsedSeconds()
                << " seconds.";
      cache_info_->Push(bytes);
    }
  }

  virtual void Fetch() = 0;

 public:
  SparsePageSourceImpl(float missing, int nthreads, bst_feature_t n_features,
                       std::shared_ptr<Cache> cache)
      : workers_{StringView{"ext-mem"}, std::max(2, std::min(nthreads, 16)), InitNewThread{}},
        missing_{missing},
        nthreads_{nthreads},
        n_features_{n_features},
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
  // Call this at the last iteration (it == n_batches).
  virtual void EndIter() {
    this->cache_info_->Commit();
    if (this->cache_info_->Size() != 0) {
      CHECK_EQ(this->count_, this->cache_info_->Size());
    }
    CHECK_GE(this->count_, 1);
    this->count_ = 0;
  }

  virtual void Reset(BatchParam const& param) {
    TryLockGuard guard{single_threaded_};

    auto at_end = false;
    std::swap(this->at_end_, at_end);

    bool changed = this->param_.n_prefetch_batches != param.n_prefetch_batches;
    this->param_ = param;

    this->count_ = 0;

    if (!at_end || changed) {
      // The last iteration did not get to the end, clear the ring to start from 0.
      this->ring_ = std::make_unique<Ring>();
    }
    this->Fetch();  // Get the 0^th page, prefetch the next page.
  }

  [[nodiscard]] auto FetchCount() const { return this->fetch_cnt_; }
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
  // Total number of batches.
  bst_idx_t n_batches_{0};

  void Fetch() final {
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
      this->n_batches_++;
      this->WriteCache();
    }
  }

 public:
  SparsePageSource(DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext> iter,
                   DMatrixProxy* proxy, float missing, int nthreads, bst_feature_t n_features,
                   bst_idx_t n_batches, std::shared_ptr<Cache> cache)
      : SparsePageSourceImpl(missing, nthreads, n_features, cache),
        iter_{iter},
        proxy_{proxy},
        n_batches_{n_batches} {
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
    } else {
      this->Fetch();
    }

    return *this;
  }

  void Reset(BatchParam const& param) override {
    if (proxy_) {
      TryLockGuard guard{single_threaded_};
      iter_.Reset();
    }
    SparsePageSourceImpl::Reset(param);

    TryLockGuard guard{single_threaded_};
    this->base_row_id_ = 0;
  }
};

/**
 * @brief A mixin for advancing the iterator with a sparse page source.
 */
template <typename S,
          typename FormatCreatePolicy = DefaultFormatStreamPolicy<S, DefaultFormatPolicy>>
class PageSourceIncMixIn : public SparsePageSourceImpl<S, FormatCreatePolicy> {
 protected:
  std::shared_ptr<SparsePageSource> source_;
  using Super = SparsePageSourceImpl<S, FormatCreatePolicy>;
  // synchronize the row page, `hist` and `gpu_hist` don't need the original sparse page
  // so we avoid fetching it.
  bool const sync_;
  // Total number of batches.
  bst_idx_t const n_batches_{0};

 public:
  PageSourceIncMixIn(float missing, std::int32_t nthreads, bst_feature_t n_features,
                     bst_idx_t n_batches, std::shared_ptr<Cache> cache, bool sync)
      : Super::SparsePageSourceImpl{missing, nthreads, n_features, cache},
        sync_{sync},
        n_batches_{n_batches} {}
  // This function always operate on the source first, then the downstream. The downstream
  // can assume the source to be ready.
  [[nodiscard]] PageSourceIncMixIn& operator++() final {
    TryLockGuard guard{this->single_threaded_};

    // Increment the source.
    if (this->sync_) {
      ++(*source_);
    }
    // Increment self.
    ++this->count_;
    // Set at end.
    this->at_end_ = this->count_ == this->n_batches_;

    if (this->at_end_) {
      this->EndIter();
      CHECK(this->cache_info_->written);
      if (!this->sync_) {
        source_.reset();  // Make sure no unnecessary fetch.
      }
    } else {
      this->Fetch();
    }

    if (this->sync_) {
      // Sanity check.
      CHECK_EQ(source_->Iter(), this->count_);
    }
    return *this;
  }

  void Reset(BatchParam const& param) final {
    if (this->sync_ || !this->cache_info_->written) {
      this->source_->Reset(param);
    }
    Super::Reset(param);
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

/**
 * @brief operator++ implementation for ExtMemQDM.
 */
template <typename S, typename FormatCreatePolicy>
class ExtQantileSourceMixin : public SparsePageSourceImpl<S, FormatCreatePolicy> {
 protected:
  std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>> source_;
  using Super = SparsePageSourceImpl<S, FormatCreatePolicy>;

 public:
  ExtQantileSourceMixin(
      float missing, std::int32_t n_threads, bst_feature_t n_features,
      std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>> source,
      std::shared_ptr<Cache> cache)
      : Super::SparsePageSourceImpl{missing, n_threads, n_features, cache},
        source_{std::move(source)} {}
  // This function always operate on the source first, then the downstream. The downstream
  // can assume the source to be ready.
  [[nodiscard]] ExtQantileSourceMixin& operator++() final {
    TryLockGuard guard{this->single_threaded_};
    // Increment self.
    ++this->count_;
    // Set at end.
    if (this->cache_info_->written) {
      this->at_end_ = (this->Iter() == this->cache_info_->Size());
    } else {
      CHECK(this->source_);
      this->at_end_ = !this->source_->Next();
    }

    if (this->at_end_) {
      this->EndIter();

      CHECK(this->cache_info_->written);
      source_.reset();  // release the source
    } else {
      this->Fetch();
    }

    return *this;
  }

  void Reset(BatchParam const& param) final {
    if (this->source_) {
      this->source_->Reset();
    }
    Super::Reset(param);
  }
};
}  // namespace xgboost::data
#endif  // XGBOOST_DATA_SPARSE_PAGE_SOURCE_H_
