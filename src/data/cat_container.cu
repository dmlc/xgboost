/**
 * Copyright 2025, XGBoost Contributors
 */
#include <thrust/copy.h>  // for copy

#include <memory>  // for make_unique
#include <vector>  // for vector

#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/device_helpers.cuh"  // for ToSpan
#include "../common/device_vector.cuh"   // for device_vector
#include "../common/type.h"              // for GetValueT
#include "../encoder/ordinal.cuh"        // for SortNames
#include "../encoder/ordinal.h"          // for DictionaryView
#include "../encoder/types.h"            // for Overloaded
#include "cat_container.cuh"             // for CatStrArray
#include "cat_container.h"               // for CatContainer
#include "xgboost/span.h"                // for Span

namespace xgboost {
namespace cuda_impl {
struct CatContainerImpl {
  std::vector<ColumnType> columns;
  dh::device_vector<enc::DeviceCatIndexView> columns_v;

  template <typename VariantT>
  void CopyFrom(Context const* ctx, enc::detail::ColumnsViewImpl<VariantT> that) {
    this->columns.resize(that.columns.size());
    this->columns_v.resize(that.columns.size());
    CHECK_EQ(this->columns.size(), this->columns_v.size());
    auto stream = ctx->CUDACtx()->Stream();

    std::vector<decltype(columns_v)::value_type> h_columns_v(this->columns_v.size());
    for (std::size_t f_idx = 0, n = that.columns.size(); f_idx < n; ++f_idx) {
      auto const& col_v = that.columns[f_idx];
      auto dispatch = enc::Overloaded{
          [this, f_idx, &h_columns_v, stream](enc::CatStrArrayView const& str) {
            this->columns[f_idx].emplace<CatStrArray>();
            auto& col = std::get<CatStrArray>(this->columns[f_idx]);
            // Handle the offsets
            col.offsets.resize(str.offsets.size());
            if (!str.offsets.empty()) {
              dh::safe_cuda(cudaMemcpyAsync(thrust::raw_pointer_cast(col.offsets.data()),
                                            str.offsets.data(), str.offsets.size_bytes(),
                                            cudaMemcpyDefault, stream));
            }
            // Handle the values
            col.values.resize(str.values.size());
            if (!col.values.empty()) {
              dh::safe_cuda(cudaMemcpyAsync(thrust::raw_pointer_cast(col.values.data()),
                                            str.values.data(), str.values.size_bytes(),
                                            cudaMemcpyDefault, stream));
            }
            // Create the view
            h_columns_v[f_idx].emplace<enc::CatStrArrayView>();
            auto& col_v = cuda::std::get<enc::CatStrArrayView>(h_columns_v[f_idx]);
            col_v = {dh::ToSpan(col.offsets), dh::ToSpan(col.values)};
          },
          [this, f_idx, &h_columns_v, stream](auto&& values) {
            using T = std::remove_cv_t<typename std::decay_t<decltype(values)>::value_type>;

            this->columns[f_idx].emplace<dh::device_vector<T>>();
            auto& col = std::get<dh::device_vector<T>>(this->columns[f_idx]);

            col.resize(values.size());
            if (!values.empty()) {
              dh::safe_cuda(cudaMemcpyAsync(col.data().get(), values.data(), values.size_bytes(),
                                            cudaMemcpyDefault, stream));
            }

            // Create the view
            using V = common::Span<std::add_const_t<T>>;
            h_columns_v[f_idx].emplace<V>();
            auto& col_v = cuda::std::get<V>(h_columns_v[f_idx]);
            col_v = dh::ToSpan(col);
          }};
      auto visit = [&](auto const& col) {
        using ColT = common::GetValueT<decltype(col)>;
        if constexpr (std::is_same_v<ColT, enc::HostCatIndexView>) {
          std::visit(dispatch, col);
        } else {
          static_assert(std::is_same_v<ColT, enc::DeviceCatIndexView>);
          cuda::std::visit(dispatch, col);
        }
      };
      visit(col_v);
    }
    thrust::copy_n(h_columns_v.data(), h_columns_v.size(), this->columns_v.data());

    CHECK_EQ(this->columns.size(), this->columns_v.size());
  }

  void CopyTo(cpu_impl::CatContainerImpl* that) {
    CHECK_EQ(this->columns.size(), this->columns_v.size());
    that->columns.clear();
    for (auto const& col : this->columns) {
      that->columns.emplace_back();
      auto& out_col = that->columns.back();

      std::visit(enc::Overloaded{
                     [&](CatStrArray const& str) {
                       out_col.emplace<cpu_impl::CatStrArray>();
                       auto& out_str = std::get<cpu_impl::CatStrArray>(out_col);
                       // Offsets
                       out_str.offsets.resize(str.offsets.size());
                       if (!out_str.offsets.empty()) {
                         dh::safe_cuda(cudaMemcpyAsync(
                             out_str.offsets.data(), thrust::raw_pointer_cast(str.offsets.data()),
                             common::Span{out_str.offsets}.size_bytes(), cudaMemcpyDefault));
                       }
                       // Values
                       out_str.values.resize(str.values.size());
                       if (!out_str.values.empty()) {
                         dh::safe_cuda(cudaMemcpyAsync(
                             out_str.values.data(), thrust::raw_pointer_cast(str.values.data()),
                             common::Span{out_str.values}.size_bytes(), cudaMemcpyDefault));
                       }
                     },
                     [&](auto&& values) {
                       using T0 = decltype(values);
                       using T1 = std::add_const_t<typename std::decay_t<T0>::value_type>;
                       using Vec = typename cpu_impl::ViewToStorageImpl<common::Span<T1>>::Type;
                       out_col.emplace<Vec>();
                       auto& out_vec = std::get<Vec>(out_col);
                       out_vec.resize(values.size());
                       if (!out_vec.empty()) {
                         dh::safe_cuda(cudaMemcpyAsync(
                             out_vec.data(), thrust::raw_pointer_cast(values.data()),
                             common::Span{out_vec}.size_bytes(), cudaMemcpyDefault));
                       }
                     }},
                 col);
    }
    that->Finalize();
  }
};

[[nodiscard]] std::tuple<CatAccessor, dh::DeviceUVector<std::int32_t>> MakeCatAccessor(
    Context const* ctx, enc::DeviceColumnsView const& new_enc, CatContainer const* orig_cats) {
  dh::DeviceUVector<std::int32_t> mapping(new_enc.n_total_cats);
  auto d_sorted_idx = orig_cats->RefSortedIndex(ctx);
  auto orig_enc = orig_cats->DeviceView(ctx);
  enc::Recode(EncPolicy, orig_enc, d_sorted_idx, new_enc, dh::ToSpan(mapping));
  CHECK_EQ(new_enc.feature_segments.size(), orig_enc.feature_segments.size());
  auto cats_mapping = enc::MappingView{new_enc.feature_segments, dh::ToSpan(mapping)};
  auto acc = CatAccessor{cats_mapping};
  return std::tuple{acc, std::move(mapping)};
}
}  // namespace cuda_impl

CatContainer::CatContainer()  // NOLINT
    : cpu_impl_{std::make_unique<cpu_impl::CatContainerImpl>()},
      cu_impl_{std::make_unique<cuda_impl::CatContainerImpl>()} {}

CatContainer::CatContainer(Context const* ctx, enc::DeviceColumnsView const& df, bool is_ref)
    : CatContainer{} {
  this->is_ref_ = is_ref;
  this->n_total_cats_ = df.n_total_cats;

  this->feature_segments_.SetDevice(ctx->Device());
  this->feature_segments_.Resize(df.feature_segments.size());
  auto d_segs = this->feature_segments_.DeviceSpan();
  thrust::copy_n(ctx->CUDACtx()->CTP(), dh::tcbegin(df.feature_segments),
                 df.feature_segments.size(), dh::tbegin(d_segs));

  // FIXME(jiamingy): We can use a single kernel for copying data once cuDF can return
  // device data. Remove this along with the one in the device cuDF adapter.
  this->cu_impl_->CopyFrom(ctx, df);

  this->sorted_idx_.SetDevice(ctx->Device());
  this->sorted_idx_.Resize(0);
  if (this->n_total_cats_ > 0) {
    CHECK(this->DeviceCanRead());
    CHECK(!this->HostCanRead());
    CHECK(!this->cu_impl_->columns.empty());
  }
}

CatContainer::~CatContainer() = default;

void CatContainer::Copy(Context const* ctx, CatContainer const& that) {
  if (ctx->IsCPU()) {
    // Pull data to host
    [[maybe_unused]] auto h_view = that.HostView();
    this->CopyCommon(ctx, that);
    this->cpu_impl_->Copy(that.cpu_impl_.get());
    CHECK(!this->DeviceCanRead());
  } else {
    // Pull data to device
    [[maybe_unused]] auto d_view = that.DeviceView(ctx);
    this->CopyCommon(ctx, that);
    auto const& that_impl = that.cu_impl_;
    this->cu_impl_->columns.resize(that.cu_impl_->columns.size());

    std::vector<decltype(this->cu_impl_->columns_v)::value_type> h_columns_v(
        that.cu_impl_->columns_v.size());
    for (std::size_t f_idx = 0, n = that_impl->columns.size(); f_idx < n; ++f_idx) {
      auto const& col = that_impl->columns[f_idx];
      std::visit(enc::Overloaded{
                     [&](cuda_impl::CatStrArray const& str) {
                       this->cu_impl_->columns[f_idx].emplace<cuda_impl::CatStrArray>();
                       auto& col = std::get<cuda_impl::CatStrArray>(this->cu_impl_->columns[f_idx]);
                       col.Copy(str);

                       h_columns_v[f_idx].emplace<enc::CatStrArrayView>();
                       auto& col_v = cuda::std::get<enc::CatStrArrayView>(h_columns_v[f_idx]);
                       col_v = {dh::ToSpan(col.offsets), dh::ToSpan(col.values)};
                     },
                     [&](auto&& values) {
                       using Vec = std::decay_t<decltype(values)>;
                       using T = typename Vec::value_type;
                       this->cu_impl_->columns[f_idx].emplace<Vec>();
                       this->cu_impl_->columns[f_idx] = values;

                       using S = common::Span<std::add_const_t<T>>;
                       h_columns_v[f_idx].emplace<S>();
                       auto& col_v = cuda::std::get<S>(h_columns_v[f_idx]);
                       col_v = dh::ToSpan(values);
                     }},
                 col);
    }
    this->cu_impl_->columns_v = h_columns_v;
    CHECK(this->Empty() || !this->HostCanRead());
  }
  if (ctx->IsCPU()) {
    CHECK_EQ(this->cpu_impl_->columns_v.size(), that.cpu_impl_->columns_v.size());
    CHECK_EQ(this->cpu_impl_->columns.size(), that.cpu_impl_->columns.size());
    CHECK(this->HostCanRead());
  } else {
    CHECK_EQ(this->cu_impl_->columns_v.size(), that.cu_impl_->columns_v.size());
    CHECK_EQ(this->cu_impl_->columns.size(), that.cu_impl_->columns.size());
    CHECK(this->DeviceCanRead());
  }
  CHECK_EQ(this->Empty(), that.Empty());
  CHECK_EQ(this->NumCatsTotal(), that.NumCatsTotal());
}

[[nodiscard]] bool CatContainer::Empty() const {
  return this->HostCanRead() ? this->cpu_impl_->columns.empty() : this->cu_impl_->columns.empty();
}

[[nodiscard]] std::size_t CatContainer::NumFeatures() const {
  if (this->HostCanRead()) {
    return this->cpu_impl_->columns.size();
  }
  return this->cu_impl_->columns.size();
}

void CatContainer::Sort(Context const* ctx) {
  if (!this->HasCategorical()) {
    return;
  }

  if (ctx->IsCPU()) {
    auto view = this->HostView();
    CHECK(!view.Empty()) << view.n_total_cats;
    this->sorted_idx_.HostVector().resize(view.n_total_cats);
    enc::SortNames(cpu_impl::EncPolicy, view, this->sorted_idx_.HostSpan());
  } else {
    auto view = this->DeviceView(ctx);
    CHECK(!view.Empty()) << view.n_total_cats;
    this->sorted_idx_.SetDevice(ctx->Device());
    this->sorted_idx_.Resize(view.n_total_cats);
    enc::SortNames(cuda_impl::EncPolicy, view, this->sorted_idx_.DeviceSpan());
  }
}

[[nodiscard]] enc::HostColumnsView CatContainer::HostView() const {
  std::lock_guard guard{device_mu_};
  if (!this->HostCanRead()) {
    this->feature_segments_.ConstHostSpan();
    // Lazy copy to host
    this->cu_impl_->CopyTo(this->cpu_impl_.get());
  }
  CHECK(this->HostCanRead());
  return this->HostViewImpl();
}

[[nodiscard]] enc::DeviceColumnsView CatContainer::DeviceView(Context const* ctx) const {
  CHECK(ctx->IsCUDA());
  std::lock_guard guard{device_mu_};
  if (!this->DeviceCanRead()) {
    this->feature_segments_.SetDevice(ctx->Device());
    this->feature_segments_.ConstDeviceSpan();
    // Lazy copy to device
    auto h_view = this->HostViewImpl();
    this->cu_impl_->CopyFrom(ctx, h_view);
    CHECK_EQ(this->cu_impl_->columns_v.size(), this->cpu_impl_->columns_v.size());
    CHECK_EQ(this->cu_impl_->columns.size(), this->cpu_impl_->columns.size());
  }
  CHECK(this->DeviceCanRead());
  if (this->n_total_cats_ != 0) {
    CHECK(!this->cu_impl_->columns_v.empty());
    CHECK_EQ(this->feature_segments_.Size(), this->cu_impl_->columns_v.size() + 1);
  }
  return {dh::ToSpan(this->cu_impl_->columns_v), this->feature_segments_.ConstDeviceSpan(),
          this->n_total_cats_};
}
}  // namespace xgboost
