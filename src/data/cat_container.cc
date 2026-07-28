/**
 * Copyright 2025-2026, XGBoost Contributors
 */
#include "cat_container.h"

#include <algorithm>    // for copy
#include <cstddef>      // for size_t
#include <cstring>      // for memcpy
#include <limits>       // for numeric_limits
#include <memory>       // for make_unique
#include <type_traits>  // for is_same_v
#include <utility>      // for move
#include <vector>       // for vector

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
// These IDs are part of the serialized category schema and must remain stable.
enum class CatIndexType : std::int64_t {
  kF32 = 7,
  kF64 = 8,
  kI8 = 9,
  kU8 = 10,
  kI16 = 11,
  kU16 = 12,
  kI32 = 13,
  kU32 = 14,
  kI64 = 15,
  kU64 = 16,
};

template <typename T>
struct CatToJson;

template <typename JsonArrayT, CatIndexType category_type>
struct CatToJsonImpl {
  using JsonArray = JsonArrayT;
  static constexpr CatIndexType kCategoryType{category_type};
};

template <>
struct CatToJson<std::uint8_t> : CatToJsonImpl<U8Array, CatIndexType::kU8> {};
template <>
struct CatToJson<std::int8_t> : CatToJsonImpl<I8Array, CatIndexType::kI8> {};
template <>
struct CatToJson<std::int16_t> : CatToJsonImpl<I16Array, CatIndexType::kI16> {};
template <>
struct CatToJson<std::int32_t> : CatToJsonImpl<I32Array, CatIndexType::kI32> {};
template <>
struct CatToJson<std::int64_t> : CatToJsonImpl<I64Array, CatIndexType::kI64> {};
template <>
struct CatToJson<std::uint16_t> : CatToJsonImpl<U16Array, CatIndexType::kU16> {};
template <>
struct CatToJson<std::uint32_t> : CatToJsonImpl<U32Array, CatIndexType::kU32> {};
template <>
struct CatToJson<std::uint64_t> : CatToJsonImpl<U64Array, CatIndexType::kU64> {};

template <typename In, typename Out>
void CopyBitPattern(std::vector<In> const& in, std::vector<Out>* out) {
  static_assert(sizeof(In) == sizeof(Out));
  out->resize(in.size());
  if (!in.empty()) {
    std::memcpy(out->data(), in.data(), in.size() * sizeof(In));
  }
}
}  // anonymous namespace

void CatContainer::Save(Json* p_out) const {
  [[maybe_unused]] auto _ = this->HostView();
  auto& out = *p_out;

  auto const& columns = this->cpu_impl_->columns;
  std::vector<Json> arr(this->cpu_impl_->columns.size());
  for (std::size_t fidx = 0, n_features = columns.size(); fidx < n_features; ++fidx) {
    auto& f_out = arr[fidx];

    auto const& col = columns[fidx];
    std::visit(
        enc::Overloaded{
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
              using T = std::remove_cv_t<typename std::decay_t<decltype(values)>::value_type>;
              using Serialization = CatToJson<T>;
              if constexpr (std::is_same_v<T, std::uint64_t>) {
                auto valid = std::all_of(values.cbegin(), values.cend(), [](T value) {
                  return value <=
                         static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
                });
                CHECK(valid) << "Category index values must not exceed the signed 64-bit range.";
              }
              using JsonArray = typename Serialization::JsonArray;
              JsonArray array{values.size()};
              auto& serialized = array.GetArray();
              std::copy(values.cbegin(), values.cend(), serialized.begin());

              Object out{};
              out["type"] = static_cast<std::int64_t>(Serialization::kCategoryType);
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
// UBJSON only supports uint8; wider unsigned arrays use same-width signed storage.
template <typename T>
using UbjUnsignedStorageT =
    std::conditional_t<(std::is_unsigned_v<T> && sizeof(T) > 1), std::make_signed_t<T>, T>;

// Dispatch method for JSON and UBJSON
template <typename U, typename Vec>
void LoadJson(Json jvalues, Vec* p_out) {
  std::vector<U> buf;
  if (IsA<Array>(jvalues)) {  // JSON
    auto const& jarray = get<Array const>(jvalues);
    buf.resize(jarray.size());
    for (std::size_t i = 0, n = jarray.size(); i < n; ++i) {
      buf[i] = static_cast<U>(get<Integer const>(jarray[i]));
    }
  } else {  // UBJSON
    using JsonArray = typename CatToJson<U>::JsonArray;
    if (IsA<JsonArray>(jvalues)) {  // Matches
      auto const& values = get<JsonArray const>(jvalues);
      buf.assign(values.cbegin(), values.cend());
    } else {  // Unsigned, needs to bit cast
      using UBJArray = typename CatToJson<UbjUnsignedStorageT<U>>::JsonArray;
      CHECK((std::is_unsigned_v<U> && !std::is_same_v<U, std::uint8_t>));
      auto const& values = get<UBJArray const>(jvalues);
      CopyBitPattern(values, &buf);
    }
  }
  *p_out = std::move(buf);
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
      auto const& jvalues = column.at("values");
      columns.emplace_back();
      switch (static_cast<CatIndexType>(type)) {
        case CatIndexType::kI8: {
          LoadJson<std::int8_t>(jvalues, &columns.back());
          break;
        }
        case CatIndexType::kU8: {
          LoadJson<std::uint8_t>(jvalues, &columns.back());
          break;
        }
        case CatIndexType::kI16: {
          LoadJson<std::int16_t>(jvalues, &columns.back());
          break;
        }
        case CatIndexType::kU16: {
          LoadJson<std::uint16_t>(jvalues, &columns.back());
          break;
        }
        case CatIndexType::kI32: {
          LoadJson<std::int32_t>(jvalues, &columns.back());
          break;
        }
        case CatIndexType::kU32: {
          LoadJson<std::uint32_t>(jvalues, &columns.back());
          break;
        }
        case CatIndexType::kI64: {
          LoadJson<std::int64_t>(jvalues, &columns.back());
          break;
        }
        case CatIndexType::kU64: {
          LoadJson<std::uint64_t>(jvalues, &columns.back());
          break;
        }
        case CatIndexType::kF32:
        case CatIndexType::kF64: {
          LOG(FATAL) << error::NoFloatCat();
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
