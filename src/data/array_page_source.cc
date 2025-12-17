/**
 * Copyright 2025, XGBoost Contributors
 */
#include "array_page_source.h"

#include "../common/cuda_rt_utils.h"

namespace xgboost::data {
ArrayPageReader::ArrayPageReader(Context const* ctx, std::uint64_t offset_bytes,
                                 std::shared_ptr<ArrayPage> cache)
    : ctx_{ctx}, cache_{std::move(cache)} {
  CHECK(cache_);
  auto const& batch_ptr = this->cache_->batch_ptr;
  auto size_bytes = [&](std::size_t i) {
    auto beg = batch_ptr.at(i);
    auto n_columns = this->cache_->gpairs.Shape(1);
    auto n_rows = beg;
    auto n_bytes = common::SizeBytes<GradientPair>(n_rows * n_columns);
    // Accumulated bytes
    return n_bytes;
  };
  auto beg = common::MakeIndexTransformIter(size_bytes);
  auto end = beg + batch_ptr.size();
  auto res_it = std::lower_bound(beg, beg + batch_ptr.size(), offset_bytes);
  CHECK(res_it != end) << "Seek failed";
  this->batch_idx_ = std::distance(beg, res_it);
}

namespace cuda_impl {
void ReadArrayPage(Context const* ctx, common::Span<GradientPair> d_dst,
                   common::Span<GradientPair const> h_src);
}

void ArrayPageReader::Read(ArrayPage* page) const {
  auto begin = this->cache_->batch_ptr.at(batch_idx_);
  auto end = this->cache_->batch_ptr.at(batch_idx_ + 1);
  auto h_cache =
      std::as_const(this->cache_->gpairs).Slice(linalg::Range(begin, end), linalg::All());
  CHECK(h_cache.CContiguous());
  page->gpairs.SetDevice(ctx_->Device());
  page->gpairs.Reshape(h_cache.Shape());
  auto d_dst = page->gpairs.View(ctx_->Device()).Values();
  auto h_src = h_cache.Values();
  cuda_impl::ReadArrayPage(this->ctx_, d_dst, h_src);
}

void ArrayPageSource::Fetch() {
  if (!this->ReadCache()) {
    this->page_.reset(new ArrayPage{});
    auto iter = this->Iter();
    auto offset = this->cache_->batch_ptr.at(iter);
    auto offset_bytes = common::SizeBytes<GradientPair>(offset * this->n_features_);
    auto reader = ArrayPageReader{this->ctx_, offset_bytes, this->cache_};
    reader.Read(this->page_.get());
    this->WriteCache();
  }
}

void ArrayPageSource::EndIter() {
  this->cache_info_->Commit();
  CHECK_GE(this->count_, 1);
  this->count_ = 0;
}

ArrayPageSource& ArrayPageSource::operator++() {
  ++this->count_;

  auto n_batches = this->cache_->NumBatches();
  CHECK_NE(n_batches, 0);

  this->at_end_ = (static_cast<decltype(n_batches)>(count_) == n_batches);
  if (this->at_end_) {
    this->EndIter();
  } else {
    this->Fetch();
  }
  return *this;
}

ArrayCacheWriter::ArrayCacheWriter(Context const* ctx, common::Span<std::size_t const, 2> shape)
    : cache_{std::make_shared<ArrayPage>()} {
  this->cache_->gpairs.SetDevice(ctx->Device());
  this->cache_->gpairs.Reshape(shape);
  auto h_cache = this->cache_->gpairs.HostView().Values();
  dh::safe_cuda(cudaHostRegister(h_cache.data(), h_cache.size_bytes(), cudaHostRegisterDefault));
}

void ArrayCacheWriter::Push(std::shared_ptr<ArrayPage> page) {
  CHECK(this->cache_);
  auto n = page->gpairs.Shape(0);
  CHECK_LE(this->offset_ + n, this->cache_->gpairs.Shape(0));
  workers_.Submit([page = std::move(page), offset = this->offset_, cache = this->cache_] {
    cache->batch_ptr.push_back(offset);
    auto h_cache = cache->gpairs.HostView();

    auto out = h_cache.Slice(offset, linalg::All());
    CHECK(out.CContiguous());
    auto in = page->gpairs.View(page->gpairs.Device());
    curt::MemcpyAsync(out.Values().data(), in.Values().data(), in.Values().size_bytes(),
                      curt::DefaultStream());
  });
  offset_ += n;
}

std::shared_ptr<ArrayPage> ArrayCacheWriter::Commit() {
  auto fut =
      workers_.Submit([&] { this->cache_->batch_ptr.push_back(this->cache_->gpairs.Shape(0)); });
  fut.get();  // Need to wait for the last task to ensure all prior tasks are complete.
  return std::move(this->cache_);
}
}  // namespace xgboost::data
