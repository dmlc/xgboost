/**
 * Copyright 2025, XGBoost Contributors
 */
#include "array_page_source.h"

#include "../common/cuda_rt_utils.h"

namespace xgboost::data {
void ArrayPageSource::Fetch() {
  if (!this->ReadCache()) {
    this->WriteCache();
  }
}

void ArrayPageSource::EndIter() { this->count_ = 0; }

ArrayPageSource& ArrayPageSource::operator++() { return *this; }

void ArrayCache::Clear() {
  if (this->last_.valid()) {
    this->last_.get();
  }
}

ArrayCache::ArrayCache(Context const* ctx, common::Span<std::size_t const, 2> shape) {
  this->cache_.gpairs.SetDevice(ctx->Device());
  this->cache_.gpairs.Reshape(shape);
  auto h_cache = this->cache_.gpairs.HostView().Values();
  dh::safe_cuda(cudaHostRegister(h_cache.data(), h_cache.size_bytes(), cudaHostRegisterDefault));
}

void ArrayCache::Push(std::shared_ptr<ArrayPage> page) {
  auto n = page->gpairs.Shape(0);
  auto h_cache = this->cache_.gpairs.HostView();
  CHECK_LE(this->offset_ + n, this->cache_.gpairs.Shape(0));
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

ArrayPage ArrayCache::Commit() {
  this->Clear();
  return std::move(this->cache_);
}
}  // namespace xgboost::data
