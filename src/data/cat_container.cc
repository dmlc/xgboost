/**
 * Copyright 2025, XGBoost Contributors
 */
#include "cat_container.h"

#include <algorithm>  // for copy
#include <cstddef>    // for size_t
#include <memory>     // for make_unique
#include <utility>    // for move
#include <vector>     // for vector

#include "../collective/allreduce.h"         // for Allreduce
#include "../collective/communicator-inl.h"  // for GetRank, GetWorldSize
#include "../common/error_msg.h"             // for NoFloatCat
#include "../encoder/types.h"                // for Overloaded
#include "xgboost/json.h"                    // for Json

namespace xgboost {
CatContainer::CatContainer(enc::HostColumnsView const& df, bool is_ref) : CatContainer{} {
  this->is_ref_ = is_ref;
  this->n_total_cats_ = df.n_total_cats;
  if (this->n_total_cats_ == 0) {
    return;
  }

  this->feature_segments_.Resize(df.feature_segments.size());
  auto& seg = this->feature_segments_.HostVector();
  std::copy_n(df.feature_segments.data(), df.feature_segments.size(), seg.begin());

  for (auto const& col : df.columns) {
    std::visit(enc::Overloaded{
                   [this](enc::CatStrArrayView str) {
                     using T = typename cpu_impl::ViewToStorageImpl<enc::CatStrArrayView>::Type;
                     this->cpu_impl_->columns.emplace_back();
                     this->cpu_impl_->columns.back().emplace<T>();
                     auto& v = std::get<T>(this->cpu_impl_->columns.back());
                     v.offsets.resize(str.offsets.size());
                     v.values.resize(str.values.size());
                     std::copy_n(str.offsets.data(), str.offsets.size(), v.offsets.data());
                     std::copy_n(str.values.data(), str.values.size(), v.values.data());
                   },
                   [this](auto&& values) {
                     using T =
                         typename cpu_impl::ViewToStorageImpl<std::decay_t<decltype(values)>>::Type;
                     this->cpu_impl_->columns.emplace_back();
                     using ElemT = typename T::value_type;

                     if constexpr (std::is_floating_point_v<ElemT>) {
                       LOG(FATAL) << error::NoFloatCat();
                     }

                     this->cpu_impl_->columns.back().emplace<T>();
                     auto& v = std::get<T>(this->cpu_impl_->columns.back());
                     v.resize(values.size());
                     std::copy_n(values.data(), values.size(), v.data());
                   }},
               col);
  }

  this->sorted_idx_.Resize(0);
  this->cpu_impl_->Finalize();

  CHECK(!this->DeviceCanRead());
  CHECK(this->HostCanRead());
  CHECK_EQ(this->n_total_cats_, df.feature_segments.back());
  CHECK_GE(this->n_total_cats_, 0) << "Too many categories.";
  if (this->n_total_cats_ > 0) {
    CHECK(!this->cpu_impl_->columns.empty());
  }
}

namespace {
template <typename T>
struct PrimToUbj;

template <>
struct PrimToUbj<std::uint8_t> {
  using Type = U8Array;
};
template <>
struct PrimToUbj<std::uint16_t> {
  using Type = U16Array;
};
template <>
struct PrimToUbj<std::uint32_t> {
  using Type = U32Array;
};
template <>
struct PrimToUbj<std::uint64_t> {
  using Type = U64Array;
};
template <>
struct PrimToUbj<std::int8_t> {
  using Type = I8Array;
};
template <>
struct PrimToUbj<std::int16_t> {
  using Type = I16Array;
};
template <>
struct PrimToUbj<std::int32_t> {
  using Type = I32Array;
};
template <>
struct PrimToUbj<std::int64_t> {
  using Type = I64Array;
};
template <>
struct PrimToUbj<float> {
  using Type = F32Array;
};
template <>
struct PrimToUbj<double> {
  using Type = F64Array;
};
}  // anonymous namespace

void CatContainer::Save(Json* p_out) const {
  [[maybe_unused]] auto _ = this->HostView();
  auto& out = *p_out;

  auto const& columns = this->cpu_impl_->columns;
  std::vector<Json> arr(this->cpu_impl_->columns.size());
  for (std::size_t fidx = 0, n_features = columns.size(); fidx < n_features; ++fidx) {
    auto& f_out = arr[fidx];

    auto const& col = columns[fidx];
    std::visit(enc::Overloaded{
                   [&f_out](cpu_impl::CatStrArray const& str) {
                     f_out = Object{};
                     I32Array joffsets{str.offsets.size()};
                     auto const& f_offsets = str.offsets;
                     std::copy(f_offsets.cbegin(), f_offsets.cend(), joffsets.GetArray().begin());
                     f_out["offsets"] = std::move(joffsets);

                     I8Array jnames{str.values.size()};  // fixme: uint8
                     auto const& f_names = str.values;
                     std::copy(f_names.cbegin(), f_names.cend(), jnames.GetArray().begin());
                     f_out["values"] = std::move(jnames);
                   },
                   [&f_out](auto&& values) {
                     using T =
                         std::remove_cv_t<typename std::decay_t<decltype(values)>::value_type>;
                     using JT = typename PrimToUbj<T>::Type;
                     JT array{values.size()};
                     std::copy_n(values.data(), values.size(), array.GetArray().begin());

                     Object out{};
                     out["type"] = static_cast<std::int64_t>(array.Type());
                     out["values"] = std::move(array);

                     f_out = std::move(out);
                   }},
               col);
  }

  auto jf_segments = I32Array{this->feature_segments_.Size()};
  auto const& hf_segments = this->feature_segments_.ConstHostVector();
  std::copy(hf_segments.cbegin(), hf_segments.cend(), jf_segments.GetArray().begin());

  auto jsorted_index = I32Array{this->sorted_idx_.Size()};
  auto const& h_sorted_idx = this->sorted_idx_.ConstHostVector();
  std::copy_n(h_sorted_idx.cbegin(), h_sorted_idx.size(), jsorted_index.GetArray().begin());

  out = Object{};
  out["sorted_idx"] = std::move(jsorted_index);
  out["feature_segments"] = std::move(jf_segments);
  out["enc"] = arr;
}

namespace {
// Dispatch method for JSON and UBJSON
template <typename U, typename Vec>
void LoadJson(Json jvalues, Vec* p_out) {
  if (IsA<Array>(jvalues)) {
    auto const& jarray = get<Array const>(jvalues);
    std::vector<U> buf(jarray.size());
    for (std::size_t i = 0, n = jarray.size(); i < n; ++i) {
      buf[i] = static_cast<U>(get<Integer const>(jarray[i]));
    }
    *p_out = std::move(buf);
    return;
  }
  auto const& values = get<std::add_const_t<typename PrimToUbj<U>::Type>>(jvalues);
  *p_out = std::move(values);
}
}  // namespace

void CatContainer::Load(Json const& in) {
  auto array = get<Array const>(in["enc"]);
  auto n_features = array.size();

  auto& columns = this->cpu_impl_->columns;
  for (std::size_t fidx = 0; fidx < n_features; ++fidx) {
    auto const& column = get<Object>(array[fidx]);
    auto it = column.find("offsets");
    if (it != column.cend()) {
      // str
      cpu_impl::CatStrArray str{};
      LoadJson<std::int32_t>(column.at("offsets"), &str.offsets);
      LoadJson<enc::CatCharT>(column.at("values"), &str.values);

      columns.emplace_back(str);
    } else {
      // numeric
      auto type = get<Integer const>(column.at("type"));
      using T = Value::ValueKind;
      auto const& jvalues = column.at("values");
      columns.emplace_back();
      switch (static_cast<Value::ValueKind>(type)) {
        case T::kI8Array: {
          LoadJson<std::int8_t>(jvalues, &columns.back());
          break;
        }
        case T::kU8Array: {
          LoadJson<std::uint8_t>(jvalues, &columns.back());
          break;
        }
        case T::kI16Array: {
          LoadJson<std::int16_t>(jvalues, &columns.back());
          break;
        }
        case T::kU16Array: {
          LoadJson<std::uint16_t>(jvalues, &columns.back());
          break;
        }
        case T::kI32Array: {
          LoadJson<std::int32_t>(jvalues, &columns.back());
          break;
        }
        case T::kU32Array: {
          LoadJson<std::uint32_t>(jvalues, &columns.back());
          break;
        }
        case T::kI64Array: {
          LoadJson<std::int64_t>(jvalues, &columns.back());
          break;
        }
        case T::kU64Array: {
          LoadJson<std::uint64_t>(jvalues, &columns.back());
          break;
        }
        case T::kF32Array: {
          LoadJson<float>(jvalues, &columns.back());
          break;
        }
        case T::kF64Array: {
          LoadJson<double>(jvalues, &columns.back());
          break;
        }
        default: {
          LOG(FATAL) << "Invalid type.";
        }
      }
    }
  }

  auto& hf_segments = this->feature_segments_.HostVector();
  LoadJson<std::int32_t>(in["feature_segments"], &hf_segments);
  if (hf_segments.empty()) {
    this->n_total_cats_ = 0;
  } else {
    this->n_total_cats_ = hf_segments.back();
  }

  auto& h_sorted_idx = this->sorted_idx_.HostVector();
  LoadJson<std::int32_t>(in["sorted_idx"], &h_sorted_idx);

  this->cpu_impl_->Finalize();
}

#if !defined(XGBOOST_USE_CUDA)
CatContainer::CatContainer() : cpu_impl_{std::make_unique<cpu_impl::CatContainerImpl>()} {}

CatContainer::~CatContainer() = default;

void CatContainer::Copy(Context const* ctx, CatContainer const& that) {
  [[maybe_unused]] auto h_view = that.HostView();
  this->CopyCommon(ctx, that);
  this->cpu_impl_->Copy(that.cpu_impl_.get());
}

[[nodiscard]] enc::HostColumnsView CatContainer::HostView() const { return this->HostViewImpl(); }

[[nodiscard]] bool CatContainer::Empty() const { return this->cpu_impl_->columns.empty(); }

[[nodiscard]] std::size_t CatContainer::NumFeatures() const {
  return this->cpu_impl_->columns.size();
}

void CatContainer::Sort(Context const* ctx) {
  CHECK(ctx->IsCPU());
  auto view = this->HostView();
  this->sorted_idx_.HostVector().resize(view.n_total_cats);
  enc::SortNames(enc::Policy<EncErrorPolicy>{}, view, this->sorted_idx_.HostSpan());
}
#endif  // !defined(XGBOOST_USE_CUDA)

void SyncCategories(Context const* ctx, CatContainer* cats, bool is_empty) {
  CHECK(cats);
  if (!collective::IsDistributed()) {
    return;
  }

  auto rank = collective::GetRank();
  std::vector<std::int32_t> workers(collective::GetWorldSize(), 0);
  workers[rank] = is_empty;
  collective::SafeColl(collective::Allreduce(ctx, &workers, collective::Op::kSum));
  if (cats->HasCategorical() &&
      std::any_of(workers.cbegin(), workers.cend(), [](auto v) { return v == 1; })) {
    LOG(FATAL)
        << "A worker cannot have empty input when a dataframe with categorical features is used. "
           "XGBoost cannot infer the categories if the input is empty.";
  }
}
}  // namespace xgboost
