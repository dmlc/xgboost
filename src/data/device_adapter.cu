/**
 * Copyright 2019-2025, XGBoost Contributors
 */
#include "../common/cuda_rt_utils.h"  // for SetDevice, CurrentDevice
#include "columnar.h"                 // for GetRefCats, GetArrowDictionary
#include "device_adapter.cuh"

namespace xgboost::data {
namespace {
auto GetRefCats(Context const* ctx, Json handle,
                std::vector<enc::DeviceCatIndexView>* p_h_ref_cats) {
  auto& h_ref_cats = *p_h_ref_cats;
  auto cats = reinterpret_cast<CatContainer const*>(get<Integer const>(handle));
  CHECK(cats);
  auto d_cats = cats->DeviceView(ctx);
  // FIXME(jiamingy): Remove this along with the host copy in the cat container once
  // cuDF can return device-only data.
  h_ref_cats.resize(d_cats.columns.size());
  thrust::copy(dh::tcbegin(d_cats.columns), dh::tcend(d_cats.columns), h_ref_cats.begin());
  d_cats.columns = common::Span{h_ref_cats};
  return d_cats;
}
}  // anonymous namespace

CudfAdapter::CudfAdapter(StringView cuda_arrinf) {
  Json jdf = Json::Load(cuda_arrinf);

  if (IsA<Object>(jdf)) {
    // Has reference categories.
    auto ctx = Context{}.MakeCUDA(curt::CurrentDevice());
    this->ref_cats_ = GetRefCats(&ctx, jdf["ref_categories"], &this->h_ref_cats_);
    jdf = jdf["columns"];
  }

  std::vector<Json> const& jcolumns = get<Array>(jdf);
  std::size_t n_columns = jcolumns.size();
  CHECK_GT(n_columns, 0) << "The number of columns must not equal to 0.";

  std::vector<ArrayInterface<1>> columns;
  std::vector<std::int32_t> cat_segments{0};
  std::int32_t device = -1;
  for (auto const& jcol : jcolumns) {
    std::int32_t n_cats{0};
    if (IsA<Array>(jcol)) {
      // This is a dictionary type (categorical values).
      auto const& first = get<Object const>(jcol[0]);
      if (first.find("offsets") == first.cend()) {
        // numeric index
        if (device == -1) {
          auto const& first = get<Object const>(jcol[0]);
          auto names = ArrayInterface<1>{first};
          device = dh::CudaGetPointerDevice(names.data);
        }
        n_cats = GetArrowNumericIndex(DeviceOrd::CUDA(device), jcol, &cats_, &columns, &n_bytes_,
                                      &num_rows_);
      } else {
        // string index
        n_cats = GetArrowDictionary(jcol, &cats_, &columns, &n_bytes_, &num_rows_);
      }
    } else {
      // Numeric values
      auto col = ArrayInterface<1>(get<Object const>(jcol));
      columns.push_back(col);
      this->cats_.emplace_back();
      this->num_rows_ = std::max(num_rows_, col.Shape<0>());
      CHECK_EQ(num_rows_, col.Shape<0>()) << "All columns should have the same number of rows.";
      n_bytes_ += col.ElementSize() * col.Shape<0>();
    }
    cat_segments.emplace_back(n_cats);
    if (device == -1) {
      device = dh::CudaGetPointerDevice(columns.back().data);
    }
    CHECK_EQ(device, dh::CudaGetPointerDevice(columns.back().data))
        << "All columns should use the same device.";
  }
  // Categories
  std::partial_sum(cat_segments.cbegin(), cat_segments.cend(), cat_segments.begin());
  this->n_total_cats_ = cat_segments.back();
  this->cat_segments_ = std::move(cat_segments);
  this->d_cats_ = this->cats_;  // thrust copy

  CHECK(!columns.empty());
  if (device < 0) {
    // Empty dataset
    CHECK_EQ(columns.front().Shape<0>(), 0);
    device_ = DeviceOrd::CUDA(curt::CurrentDevice());
  } else {
    device_ = DeviceOrd::CUDA(device);
  }
  CHECK(device_.IsCUDA());
  curt::SetDevice(device_.ordinal);

  this->columns_ = columns;
  batch_ = CudfAdapterBatch(dh::ToSpan(columns_), NoOpAccessor{}, num_rows_);

  if (!this->ref_cats_.Empty()) {
    CHECK_EQ(this->ref_cats_.Size(), this->columns_.size())
        << "Invalid reference categories, different number of columns";
  }
}
}  // namespace xgboost::data
