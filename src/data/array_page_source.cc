/**
 * Copyright 2025, XGBoost Contributors
 */
#include "array_page_source.h"

#include "../common/cuda_rt_utils.h"

namespace xgboost::data {
ArrayPageReader::ArrayPageReader(std::uint64_t offset_bytes,
                                 std::vector<bst_idx_t> const& batch_ptr,
                                 std::shared_ptr<ArrayPage> cache)
    : cache_{std::move(cache)} {
  CHECK(cache_);
  // fixme: binary seaerch
  auto size_bytes = [&](std::size_t i) {
    auto beg = batch_ptr.at(i);
    auto n_columns = this->cache_->gpairs.Shape(1);
    auto n_rows = beg;
    auto n_bytes = common::SizeBytes<GradientPair>(n_rows * n_columns);
    // accumulated bytes
    return n_bytes;
  };
  for (std::size_t i = 0; i < batch_ptr.size() - 1; ++i) {
    auto n_bytes = size_bytes(i);
    if (n_bytes == offset_bytes) {
      this->batch_idx = i;
      break;
    }
  }
  CHECK_NE(this->batch_idx, -1) << "Seek failed";
  this->batch_ptr_ = batch_ptr;
}

void ArrayPageReader::Read(ArrayPage* page) const {
  auto begin = this->batch_ptr_.at(batch_idx);
  auto end = this->batch_ptr_.at(batch_idx + 1);
  auto h_cache =
      std::as_const(this->cache_->gpairs).Slice(linalg::Range(begin, end), linalg::All());
  Context ctx = Context{}.MakeCUDA(0);  // fixme
  CHECK(h_cache.CContiguous());
  page->gpairs.SetDevice(ctx.Device());
  page->gpairs.Reshape(h_cache.Shape());
  auto d_dst = page->gpairs.View(ctx.Device()).Values();
  curt::MemcpyAsync(d_dst.data(), h_cache.Values().data(), d_dst.size_bytes(),
                    curt::DefaultStream());
}

void ArrayPageSource::Fetch() {
  if (!this->ReadCache()) {
    this->page_.reset(new ArrayPage{});
    auto iter = this->Iter();
    auto offset = this->batch_ptr_.at(iter);
    auto offset_bytes = common::SizeBytes<GradientPair>(offset * this->n_features_);
    CHECK(this->cache_) << this;
    auto reader = ArrayPageReader{offset_bytes, this->batch_ptr_, this->cache_};
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

  CHECK(!this->batch_ptr_.empty());
  auto n_batches = this->batch_ptr_.size() - 1;
  this->at_end_ = (count_ == n_batches);
  if (this->at_end_) {
    this->EndIter();
  } else {
    this->Fetch();
  }
  return *this;
}

void ArrayCache::Clear() {
  if (this->last_.valid()) {
    this->last_.get();
  }
}

ArrayCache::ArrayCache(Context const* ctx, common::Span<std::size_t const, 2> shape)
    : cache_{std::make_shared<ArrayPage>()} {
  this->cache_->gpairs.SetDevice(ctx->Device());
  this->cache_->gpairs.Reshape(shape);
  auto h_cache = this->cache_->gpairs.HostView().Values();
  dh::safe_cuda(cudaHostRegister(h_cache.data(), h_cache.size_bytes(), cudaHostRegisterDefault));
}

void ArrayCache::Push(std::shared_ptr<ArrayPage> page) {
  CHECK(this->cache_);
  auto n = page->gpairs.Shape(0);
  auto h_cache = this->cache_->gpairs.HostView();
  CHECK_LE(this->offset_ + n, this->cache_->gpairs.Shape(0));
  auto fut = workers_.Submit([page = std::move(page), offset = this->offset_, h_cache] {
    auto out = h_cache.Slice(offset, linalg::All());
    CHECK(out.CContiguous());
    auto in = page->gpairs.View(page->gpairs.Device());
    curt::MemcpyAsync(out.Values().data(), in.Values().data(), in.Values().size_bytes(),
                      curt::DefaultStream());
  });
  offset_ += n;
  this->Clear();
  this->last_ = std::move(fut);
}

std::shared_ptr<ArrayPage> ArrayCache::Commit() {
  this->Clear();
  return std::move(this->cache_);
}
}  // namespace xgboost::data
