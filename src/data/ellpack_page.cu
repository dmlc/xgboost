/**
 * Copyright 2019-2024, XGBoost contributors
 */
#include <thrust/binary_search.h>                       // for lower_bound,  upper_bound
#include <thrust/extrema.h>                             // for max_element
#include <thrust/iterator/counting_iterator.h>          // for make_counting_iterator
#include <thrust/iterator/transform_output_iterator.h>  // for transform_output_iterator

#include <algorithm>  // for copy
#include <limits>     // for numeric_limits
#include <utility>    // for move
#include <vector>     // for vector

#include "../common/algorithm.cuh"          // for InclusiveScan
#include "../common/categorical.h"          // for IsCat
#include "../common/cuda_context.cuh"       // for CUDAContext
#include "../common/cuda_rt_utils.h"        // for SetDevice
#include "../common/hist_util.cuh"          // for HistogramCuts
#include "../common/ref_resource_view.cuh"  // for MakeFixedVecWithCudaMalloc
#include "../common/transform_iterator.h"   // for MakeIndexTransformIter
#include "device_adapter.cuh"               // for NoInfInData
#include "ellpack_page.cuh"                 // for EllpackPageImpl
#include "ellpack_page.h"                   // for EllpackPage
#include "gradient_index.h"                 // for GHistIndexMatrix
#include "xgboost/context.h"                // for Context
#include "xgboost/data.h"                   // for DMatrix

namespace xgboost {
EllpackPage::EllpackPage() : impl_{new EllpackPageImpl{}} {}

EllpackPage::EllpackPage(Context const* ctx, DMatrix* dmat, const BatchParam& param)
    : impl_{new EllpackPageImpl{ctx, dmat, param}} {}

EllpackPage::~EllpackPage() noexcept(false) = default;

EllpackPage::EllpackPage(EllpackPage&& that) { std::swap(impl_, that.impl_); }

[[nodiscard]] bst_idx_t EllpackPage::Size() const { return impl_->Size(); }

void EllpackPage::SetBaseRowId(std::size_t row_id) { impl_->SetBaseRowId(row_id); }

[[nodiscard]] common::HistogramCuts const& EllpackPage::Cuts() const {
  CHECK(impl_);
  return impl_->Cuts();
}

[[nodiscard]] bst_idx_t EllpackPage::BaseRowId() const { return this->Impl()->base_rowid; }

// Bin each input data entry, store the bin indices in compressed form.
template <bool HasNoMissing, bool kDenseCompressed>
__global__ void CompressBinEllpackKernel(
    common::CompressedBufferWriter wr,
    common::CompressedByteT* __restrict__ buffer,  // gidx_buffer
    const size_t* __restrict__ row_ptrs,           // row offset of input data
    const Entry* __restrict__ entries,             // One batch of input data
    const float* __restrict__ cuts,                // HistogramCuts::cut_values_
    const uint32_t* __restrict__ cut_ptrs,         // HistogramCuts::cut_ptrs_
    common::Span<FeatureType const> feature_types,
    size_t base_row,  // batch_row_begin
    size_t n_rows, size_t row_stride, std::uint32_t null_gidx_value) {
  auto irow = threadIdx.x + blockIdx.x * blockDim.x;
  auto cpr_fidx = threadIdx.y + blockIdx.y * blockDim.y;  // compressed fidx
  if (irow >= n_rows || cpr_fidx >= row_stride) {
    return;
  }
  auto row_length = static_cast<decltype(cpr_fidx)>(row_ptrs[irow + 1] - row_ptrs[irow]);
  std::uint32_t bin = null_gidx_value;

  // When treating a sparse matrix as dense, we need to write null values in between valid
  // values. But we don't know where to write if the feature index is not recorded for a
  // missing value. Here we use binary search to ensure `cpr_fidx` is the same as `fidx`.
  if (kDenseCompressed && !HasNoMissing) {
    auto row_beg = entries + row_ptrs[irow] - row_ptrs[0];
    auto row_end = entries + row_ptrs[irow + 1] - row_ptrs[0];
    auto it = thrust::make_transform_iterator(thrust::make_counting_iterator(0ul),
                                              [=](std::size_t i) { return row_beg[i].index; });
    auto it_end = it + thrust::distance(row_beg, row_end);
    auto res_it = thrust::lower_bound(thrust::seq, it, it_end, cpr_fidx);
    if (res_it == it_end || cpr_fidx != *res_it) {
      wr.AtomicWriteSymbol(buffer, bin, (irow + base_row) * row_stride + cpr_fidx);
      return;
    }
    cpr_fidx = thrust::distance(it, res_it);
    SPAN_CHECK(cpr_fidx < row_length);
  }

  if (cpr_fidx < row_length) {
    // We are using sub-batch of a SparsePage, need to account for the first offset within
    // the sub-batch.
    //
    // The block.y idx is calculated using row_stride, which is the longest row. We can
    // use `compressed_fidx` to fully index the sparse page row.
    Entry entry = entries[row_ptrs[irow] - row_ptrs[0] + cpr_fidx];

    bst_feature_t fidx = entry.index;
    float fvalue = entry.fvalue;
    // {feature_cuts, n_cuts} forms the array of cuts of the current `feature'.
    float const* feature_cuts = &cuts[cut_ptrs[fidx]];
    auto n_cuts = cut_ptrs[fidx + 1] - cut_ptrs[fidx];

    bool is_cat = common::IsCat(feature_types, fidx);
    // Assigning the bin in current entry.
    // S.t.: fvalue < feature_cuts[bin]
    bin = std::numeric_limits<decltype(bin)>::max();
    if (is_cat) {
      auto it =
          dh::MakeTransformIterator<int>(feature_cuts, [](float v) { return common::AsCat(v); });
      bin = thrust::lower_bound(thrust::seq, it, it + n_cuts, common::AsCat(fvalue)) - it;
    } else {
      bin = thrust::upper_bound(thrust::seq, feature_cuts, feature_cuts + n_cuts, fvalue) -
            feature_cuts;
    }

    if (bin >= n_cuts) {
      bin = n_cuts - 1;
    }
    if (!kDenseCompressed) {
      // Sparse data, use the compressed fidx.  Add the number of bins in previous
      // features since we can't compresse it based on feature-local index.
      bin += cut_ptrs[fidx];
    } else {
      // Write to the actual fidx for dense data.
      cpr_fidx = fidx;
    }
  }
  // Write to the gidx buffer for non-missing values.
  wr.AtomicWriteSymbol(buffer, bin, (irow + base_row) * row_stride + cpr_fidx);
}

// Calculate the number of symbols for the compressed ellpack. Similar to what the CPU
// implementation does, we compress the dense data by subtracting the bin values with the
// starting bin of its feature if it's dense. In addition, we treat the data as dense if
// there's no compression to be made by using ellpack.
[[nodiscard]] EllpackPageImpl::Info CalcNumSymbols(
    Context const* ctx, bst_idx_t row_stride, bool is_dense,
    std::shared_ptr<common::HistogramCuts const> cuts) {
  // Return the total number of symbols (total number of bins plus 1 for missing)
  // The null value equals the total number of bins.
  bst_idx_t n_symbols = cuts->TotalBins() + 1;
  if (n_symbols == 1) {  // Empty DMatrix
    return {static_cast<bst_feature_t>(0), n_symbols};
  }

  bst_idx_t n_features = cuts->NumFeatures();
  cuts->cut_ptrs_.SetDevice(ctx->Device());
  common::Span<std::uint32_t const> dptrs = cuts->cut_ptrs_.ConstDeviceSpan();
  using PtrT = typename decltype(dptrs)::value_type;

  // Calculate the number of required symbols if we treat the data as dense.
  PtrT n_symbols_dense{0};
  CUDAContext const* cuctx = ctx->CUDACtx();
  auto it = dh::MakeTransformIterator<PtrT>(
      thrust::make_counting_iterator(1ul),
      [=] XGBOOST_DEVICE(std::size_t i) { return dptrs[i] - dptrs[i - 1]; });
  CHECK_GE(dptrs.size(), 2);
  auto max_it = thrust::max_element(cuctx->CTP(), it, it + dptrs.size() - 1);
  dh::CachingDeviceUVector<PtrT> max_element(1);
  auto d_me = max_element.data();
  dh::LaunchN(1, cuctx->Stream(), [=] XGBOOST_DEVICE(std::size_t i) { d_me[i] = *max_it; });
  dh::safe_cuda(cudaMemcpyAsync(&n_symbols_dense, d_me, sizeof(PtrT), cudaMemcpyDeviceToHost,
                                cuctx->Stream()));
  cuctx->Stream().Sync();
  // Decide the type of the data.
  CHECK_LE(row_stride, n_features);
  if (is_dense) {
    // No missing, hence no null value, hence no + 1 symbol.
    LOG(INFO) << "Ellpack is dense.";
    return {n_features, n_symbols_dense};
  } else if (n_features == row_stride) {
    // Treat the ellpack as dense if we can save memory.
    LOG(INFO) << "Ellpack is relatively dense.";
    return {n_features, n_symbols_dense + 1};  // +1 for missing value (null in ellpack)
  } else {
    LOG(INFO) << "Ellpack is sparse.";
    return {row_stride, n_symbols};
  }
}

// Construct an ELLPACK matrix with the given number of empty rows.
EllpackPageImpl::EllpackPageImpl(Context const* ctx,
                                 std::shared_ptr<common::HistogramCuts const> cuts, bool is_dense,
                                 bst_idx_t row_stride, bst_idx_t n_rows)
    : is_dense{is_dense},
      n_rows{n_rows},
      cuts_{std::move(cuts)},
      info{CalcNumSymbols(ctx, row_stride, is_dense, this->cuts_)} {
  monitor_.Init("ellpack_page");
  curt::SetDevice(ctx->Ordinal());

  this->InitCompressedData(ctx);
}

EllpackPageImpl::EllpackPageImpl(Context const* ctx,
                                 std::shared_ptr<common::HistogramCuts const> cuts,
                                 const SparsePage& page, bool is_dense, size_t row_stride,
                                 common::Span<FeatureType const> feature_types)
    : is_dense{is_dense},
      n_rows{page.Size()},
      cuts_{std::move(cuts)},
      info{CalcNumSymbols(ctx, row_stride, is_dense, this->cuts_)} {
  monitor_.Init("ellpack_page");
  curt::SetDevice(ctx->Ordinal());

  this->InitCompressedData(ctx);
  this->CreateHistIndices(ctx, page, feature_types);
}

// Construct an ELLPACK matrix in memory.
EllpackPageImpl::EllpackPageImpl(Context const* ctx, DMatrix* p_fmat, const BatchParam& param)
    : is_dense{p_fmat->IsDense()},
      n_rows{p_fmat->Info().num_row_},
      // Create the quantile sketches for the dmatrix and initialize HistogramCuts.
      cuts_{param.hess.empty()
                ? std::make_shared<common::HistogramCuts>(
                      common::DeviceSketch(ctx, p_fmat, param.max_bin))
                : std::make_shared<common::HistogramCuts>(
                      common::DeviceSketchWithHessian(ctx, p_fmat, param.max_bin, param.hess))},
      info{CalcNumSymbols(ctx, GetRowStride(p_fmat), p_fmat->IsDense(), this->cuts_)} {
  monitor_.Init("ellpack_page");
  curt::SetDevice(ctx->Ordinal());

  this->InitCompressedData(ctx);

  p_fmat->Info().feature_types.SetDevice(ctx->Device());
  auto ft = p_fmat->Info().feature_types.ConstDeviceSpan();
  CHECK(p_fmat->SingleColBlock());
  for (auto const& page : p_fmat->GetBatches<SparsePage>()) {
    this->CreateHistIndices(ctx, page, ft);
  }
}

template <typename AdapterBatchT>
struct WriteCompressedEllpackFunctor {
  WriteCompressedEllpackFunctor(common::CompressedByteT* buffer,
                                const common::CompressedBufferWriter& writer, AdapterBatchT batch,
                                EllpackDeviceAccessor accessor,
                                common::Span<FeatureType const> feature_types,
                                const data::IsValidFunctor& is_valid)
      : d_buffer(buffer),
        writer(writer),
        batch(std::move(batch)),
        accessor(std::move(accessor)),
        feature_types(std::move(feature_types)),
        is_valid(is_valid) {}

  common::CompressedByteT* d_buffer;
  common::CompressedBufferWriter writer;
  AdapterBatchT batch;
  EllpackDeviceAccessor accessor;
  common::Span<FeatureType const> feature_types;
  data::IsValidFunctor is_valid;

  // Tuple[0] = The row index of the input, used as a key to define segments
  // Tuple[1] = Scanned flags of valid elements for each row
  // Tuple[2] = The index in the input data
  using Tuple = thrust::tuple<bst_idx_t, bst_idx_t, bst_idx_t>;

  template <bool kIsDenseCompressed>
  __device__ void Write(data::COOTuple const& e, bst_idx_t out_position) {
    bst_bin_t bin_idx = 0;
    if (common::IsCat(feature_types, e.column_idx)) {
      bin_idx = accessor.SearchBin<true>(e.value, e.column_idx);
    } else {
      bin_idx = accessor.SearchBin<false>(e.value, e.column_idx);
    }
    if constexpr (kIsDenseCompressed) {
      bin_idx -= accessor.feature_segments[e.column_idx];
    }
    writer.AtomicWriteSymbol(d_buffer, bin_idx, out_position);
  }
  // Used for dense or as dense data.
  __device__ void operator()(bst_idx_t i) {
    auto e = batch.GetElement(i);
    if (is_valid(e)) {
      this->Write<true>(e, i);
    } else {
      writer.AtomicWriteSymbol(d_buffer, accessor.NullValue(), i);
    }
  }
  // Used for sparse data.
  __device__ size_t operator()(Tuple const& out) {
    auto e = batch.GetElement(thrust::get<2>(out));
    if (is_valid(e)) {
      // -1 because the scan is inclusive
      size_t output_position = accessor.row_stride * e.row_idx + thrust::get<1>(out) - 1;
      this->Write<false>(e, output_position);
    }
    return 0;
  }
};

template <typename Tuple>
struct TupleScanOp {
  __device__ Tuple operator()(Tuple a, Tuple b) {
    // Key equal
    if (thrust::get<0>(a) == thrust::get<0>(b)) {
      thrust::get<1>(b) += thrust::get<1>(a);
      return b;
    }
    // Not equal
    return b;
  }
};

// Here the data is already correctly ordered and simply needs to be compacted
// to remove missing data
template <bool kIsDenseCompressed, typename AdapterBatchT>
void CopyDataToEllpack(Context const* ctx, const AdapterBatchT& batch,
                       common::Span<FeatureType const> feature_types, EllpackPageImpl* dst,
                       float missing) {
  data::IsValidFunctor is_valid(missing);
  bool valid = data::NoInfInData(ctx, batch, is_valid);
  CHECK(valid) << error::InfInData();

  auto cnt = thrust::make_counting_iterator(0llu);
  auto n_symbols = dst->NumSymbols();
  common::CompressedBufferWriter writer{n_symbols};
  auto d_compressed_buffer = dst->gidx_buffer.data();

  // We redirect the scan output into this functor to do the actual writing
  using Tuple = typename WriteCompressedEllpackFunctor<AdapterBatchT>::Tuple;
  dh::TypedDiscard<Tuple> discard;
  auto device_accessor = dst->GetDeviceAccessor(ctx);
  WriteCompressedEllpackFunctor<AdapterBatchT> functor{
      d_compressed_buffer, writer, batch, device_accessor, feature_types, is_valid};

  // For dense compressed data, we can simply copy the data with the input position.
  if (kIsDenseCompressed) {
    CHECK(batch.NumRows() == 0 || batch.NumCols() == dst->info.row_stride);
    thrust::for_each_n(ctx->CUDACtx()->CTP(), cnt, dst->Size() * dst->info.row_stride, functor);
    return;
  }

  // Some witchcraft happens here.
  //
  // The goal is to copy valid elements out of the input to an ELLPACK matrix with a given
  // row stride, using no extra working memory Standard stream compaction needs to be
  // modified to do this, so we manually define a segmented stream compaction via
  // operators on an inclusive scan. The output of this inclusive scan is fed to a custom
  // function which works out the correct output position
  auto key_iter = dh::MakeTransformIterator<size_t>(
      cnt, [=] __device__(size_t idx) { return batch.GetElement(idx).row_idx; });
  auto value_iter = dh::MakeTransformIterator<size_t>(
      cnt, [=] __device__(size_t idx) -> size_t { return is_valid(batch.GetElement(idx)); });

  auto key_value_index_iter =
      thrust::make_zip_iterator(thrust::make_tuple(key_iter, value_iter, cnt));
  thrust::transform_output_iterator<decltype(functor), decltype(discard)> out(discard, functor);
  common::InclusiveScan(ctx, key_value_index_iter, out, TupleScanOp<Tuple>{}, batch.Size());
}

void WriteNullValues(Context const* ctx, EllpackPageImpl* dst,
                     common::Span<size_t const> row_counts) {
  // Write the null values
  auto null = dst->GetDeviceAccessor(ctx).NullValue();
  common::CompressedBufferWriter writer(dst->NumSymbols());
  auto d_compressed_buffer = dst->gidx_buffer.data();
  auto row_stride = dst->info.row_stride;
  auto n = row_stride * dst->n_rows;
  dh::LaunchN(n, ctx->CUDACtx()->Stream(), [=] __device__(bst_idx_t idx) mutable {
    size_t row_idx = idx / row_stride;
    size_t row_offset = idx % row_stride;
    if (row_offset >= row_counts[row_idx]) {
      writer.AtomicWriteSymbol(d_compressed_buffer, null, idx);
    }
  });
}

template <typename AdapterBatch>
EllpackPageImpl::EllpackPageImpl(Context const* ctx, AdapterBatch batch, float missing,
                                 bool is_dense, common::Span<bst_idx_t const> row_counts,
                                 common::Span<FeatureType const> feature_types,
                                 bst_idx_t row_stride, bst_idx_t n_rows,
                                 std::shared_ptr<common::HistogramCuts const> cuts)
    : EllpackPageImpl{ctx, cuts, is_dense, row_stride, n_rows} {
  curt::SetDevice(ctx->Ordinal());

  if (this->IsDenseCompressed()) {
    CopyDataToEllpack<true>(ctx, batch, feature_types, this, missing);
  } else {
    CopyDataToEllpack<false>(ctx, batch, feature_types, this, missing);
    WriteNullValues(ctx, this, row_counts);
  }
}

#define ELLPACK_BATCH_SPECIALIZE(__BATCH_T)                                                  \
  template EllpackPageImpl::EllpackPageImpl(                                                 \
      Context const* ctx, __BATCH_T batch, float missing, bool is_dense,                     \
      common::Span<bst_idx_t const> row_counts_span,                                         \
      common::Span<FeatureType const> feature_types, bst_idx_t row_stride, bst_idx_t n_rows, \
      std::shared_ptr<common::HistogramCuts const> cuts);

ELLPACK_BATCH_SPECIALIZE(data::CudfAdapterBatch)
ELLPACK_BATCH_SPECIALIZE(data::CupyAdapterBatch)

#undef ELLPACK_BATCH_SPECIALIZE

namespace {
template <typename T>
void CopyGHistToEllpack(Context const* ctx, GHistIndexMatrix const& page,
                        common::Span<bst_idx_t const> d_row_ptr, bst_idx_t row_stride,
                        bst_bin_t null, bst_idx_t n_symbols,
                        common::Span<bst_feature_t const> d_cut_ptrs,
                        common::CompressedByteT* d_compressed_buffer) {
  dh::device_vector<uint8_t> data(page.index.begin(), page.index.end());
  auto d_data = dh::ToSpan(data);

  // GPU employs the same dense compression as CPU, no need to handle page.index.Offset()
  auto bin_type = page.index.GetBinTypeSize();
  common::CompressedBufferWriter writer{n_symbols};
  auto cuctx = ctx->CUDACtx();

  bool dense_compress = row_stride == page.Features() && !page.IsDense();
  auto n_samples = page.Size();
  auto cnt = thrust::make_counting_iterator(0ul);
  auto ptr = reinterpret_cast<T const*>(d_data.data());
  auto fn = [=] __device__(std::size_t i) mutable {
    auto [ridx, fidx] = linalg::UnravelIndex(i, n_samples, row_stride);
    auto r_begin = d_row_ptr[ridx];
    auto r_end = d_row_ptr[ridx + 1];
    auto r_size = r_end - r_begin;

    bst_bin_t bin_idx;
    if (dense_compress) {
      auto f_begin = d_cut_ptrs[fidx];
      auto f_end = d_cut_ptrs[fidx + 1];
      // CPU gidx is not compressed, can be used for binary search.
      bin_idx = common::BinarySearchBin(r_begin, r_end, ptr, f_begin, f_end);
      if (bin_idx == -1) {
        bin_idx = null;
      } else {
        bin_idx -= d_cut_ptrs[fidx];
      }
    } else if (fidx >= r_size) {
      bin_idx = null;
    } else {
      bin_idx = ptr[r_begin + fidx];
    }

    writer.AtomicWriteSymbol(d_compressed_buffer, bin_idx, i);
  };
  thrust::for_each_n(cuctx->CTP(), cnt, row_stride * page.Size(), fn);
}
}  // anonymous namespace

EllpackPageImpl::EllpackPageImpl(Context const* ctx, GHistIndexMatrix const& page,
                                 common::Span<FeatureType const> ft)
    : is_dense{page.IsDense()},
      base_rowid{page.base_rowid},
      n_rows{page.Size()},
      cuts_{[&] {
        auto cuts = std::make_shared<common::HistogramCuts>(page.cut);
        cuts->SetDevice(ctx->Device());
        return cuts;
      }()},
      info{CalcNumSymbols(
          ctx,
          [&] {
            auto it = common::MakeIndexTransformIter(
                [&](bst_idx_t i) { return page.row_ptr[i + 1] - page.row_ptr[i]; });
            return *std::max_element(it, it + page.Size());
          }(),
          page.IsDense(), cuts_)} {
  this->monitor_.Init("ellpack_page");
  CHECK(ctx->IsCUDA());
  this->InitCompressedData(ctx);

  // copy gidx
  common::CompressedByteT* d_compressed_buffer = gidx_buffer.data();
  dh::device_vector<size_t> row_ptr(page.row_ptr.size());
  auto d_row_ptr = dh::ToSpan(row_ptr);
  dh::safe_cuda(cudaMemcpyAsync(d_row_ptr.data(), page.row_ptr.data(), d_row_ptr.size_bytes(),
                                cudaMemcpyHostToDevice, ctx->CUDACtx()->Stream()));

  auto accessor = this->GetDeviceAccessor(ctx, ft);
  this->monitor_.Start("CopyGHistToEllpack");
  common::DispatchBinType(page.index.GetBinTypeSize(), [&](auto t) {
    using T = decltype(t);
    CopyGHistToEllpack<T>(ctx, page, d_row_ptr, this->info.row_stride, accessor.NullValue(),
                          this->NumSymbols(), this->cuts_->cut_ptrs_.ConstDeviceSpan(),
                          d_compressed_buffer);
  });
  this->monitor_.Stop("CopyGHistToEllpack");
}

EllpackPageImpl::~EllpackPageImpl() noexcept(false) {
  // Sync the stream to make sure all running CUDA kernels finish before deallocation.
  auto status = dh::DefaultStream().Sync(false);
  if (status != cudaSuccess) {
    auto str = cudaGetErrorString(status);
    // For external-memory, throwing here can trigger a series of calls to
    // `std::terminate` by various destructors. For now, we just log the error.
    LOG(WARNING) << "Ran into CUDA error:" << str << "\nXGBoost is likely to abort.";
  }
  dh::safe_cuda(status);
}

// A functor that copies the data from one EllpackPage to another.
struct CopyPage {
  common::CompressedBufferWriter cbw;
  common::CompressedByteT* dst_data_d;
  common::CompressedIterator<uint32_t> src_iterator_d;
  // The number of elements to skip.
  size_t offset;

  CopyPage(EllpackPageImpl* dst, EllpackPageImpl const* src, size_t offset)
      : cbw{dst->NumSymbols()},
        dst_data_d{dst->gidx_buffer.data()},
        src_iterator_d{src->gidx_buffer.data(), src->NumSymbols()},
        offset{offset} {}

  __device__ void operator()(std::size_t element_id) {
    cbw.AtomicWriteSymbol(dst_data_d, src_iterator_d[element_id], element_id + offset);
  }
};

// Copy the data from the given EllpackPage to the current page.
bst_idx_t EllpackPageImpl::Copy(Context const* ctx, EllpackPageImpl const* page, bst_idx_t offset) {
  monitor_.Start(__func__);
  bst_idx_t n_elements = page->n_rows * page->info.row_stride;
  CHECK_NE(this, page);
  CHECK_EQ(this->info.row_stride, page->info.row_stride);
  CHECK_EQ(this->NumSymbols(), page->NumSymbols());
  CHECK_GE(this->n_rows * this->info.row_stride, offset + n_elements);
  thrust::for_each_n(ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(0ul), n_elements,
                     CopyPage{this, page, offset});
  monitor_.Stop(__func__);
  return n_elements;
}

// A functor that compacts the rows from one EllpackPage into another.
struct CompactPage {
  common::CompressedBufferWriter cbw;
  common::CompressedByteT* dst_data_d;
  common::CompressedIterator<uint32_t> src_iterator_d;
  /**
   * @brief An array that maps the rows from the full DMatrix to the compacted page.
   *
   * The total size is the number of rows in the original, uncompacted DMatrix.
   * Elements are the row ids in the compacted page. Rows not needed are set to
   * SIZE_MAX.
   *
   * An example compacting 16 rows to 8 rows:
   * [SIZE_MAX, 0, 1, SIZE_MAX, SIZE_MAX, 2, SIZE_MAX, 3, 4, 5, SIZE_MAX, 6,
   * SIZE_MAX, 7, SIZE_MAX, SIZE_MAX]
   */
  common::Span<size_t> row_indexes;
  size_t base_rowid;
  size_t row_stride;

  CompactPage(EllpackPageImpl* dst, EllpackPageImpl const* src, common::Span<size_t> row_indexes)
      : cbw{dst->NumSymbols()},
        dst_data_d{dst->gidx_buffer.data()},
        src_iterator_d{src->gidx_buffer.data(), src->NumSymbols()},
        row_indexes(row_indexes),
        base_rowid{src->base_rowid},
        row_stride{src->info.row_stride} {}

  __device__ void operator()(bst_idx_t row_id) {
    size_t src_row = base_rowid + row_id;
    size_t dst_row = row_indexes[src_row];
    if (dst_row == SIZE_MAX) {
      return;
    }
    size_t dst_offset = dst_row * row_stride;
    size_t src_offset = row_id * row_stride;
    for (size_t j = 0; j < row_stride; j++) {
      cbw.AtomicWriteSymbol(dst_data_d, src_iterator_d[src_offset + j], dst_offset + j);
    }
  }
};

// Compacts the data from the given EllpackPage into the current page.
void EllpackPageImpl::Compact(Context const* ctx, EllpackPageImpl const* page,
                              common::Span<size_t> row_indexes) {
  monitor_.Start(__func__);
  CHECK_EQ(this->info.row_stride, page->info.row_stride);
  CHECK_EQ(this->NumSymbols(), page->NumSymbols());
  CHECK_LE(page->base_rowid + page->n_rows, row_indexes.size());
  auto cuctx = ctx->CUDACtx();
  dh::LaunchN(page->n_rows, cuctx->Stream(), CompactPage{this, page, row_indexes});
  monitor_.Stop(__func__);
}

void EllpackPageImpl::SetCuts(std::shared_ptr<common::HistogramCuts const> cuts) {
  cuts_ = std::move(cuts);
}

// Initialize the buffer to stored compressed features.
void EllpackPageImpl::InitCompressedData(Context const* ctx) {
  monitor_.Start(__func__);
  auto num_symbols = this->NumSymbols();
  // Required buffer size for storing data matrix in ELLPack format.
  std::size_t compressed_size_bytes = common::CompressedBufferWriter::CalculateBufferSize(
      this->info.row_stride * this->n_rows, num_symbols);
  auto init = static_cast<common::CompressedByteT>(0);
  gidx_buffer = common::MakeFixedVecWithCudaMalloc(ctx, compressed_size_bytes, init);
  monitor_.Stop(__func__);
}

// Compress a CSR page into ELLPACK.
void EllpackPageImpl::CreateHistIndices(Context const* ctx, const SparsePage& row_batch,
                                        common::Span<FeatureType const> feature_types) {
  if (row_batch.Size() == 0) {
    return;
  }

  this->monitor_.Start(__func__);
  auto null_gidx_value = this->GetDeviceAccessor(ctx, feature_types).NullValue();

  auto const& offset_vec = row_batch.offset.ConstHostVector();

  // bin and compress entries in batches of rows
  size_t gpu_batch_nrows =
      std::min(curt::TotalMemory() / (16 * this->info.row_stride * sizeof(Entry)),
               static_cast<size_t>(row_batch.Size()));

  size_t gpu_nbatches = common::DivRoundUp(row_batch.Size(), gpu_batch_nrows);
  auto writer = common::CompressedBufferWriter{this->NumSymbols()};
  auto gidx_buffer_data = gidx_buffer.data();

  for (size_t gpu_batch = 0; gpu_batch < gpu_nbatches; ++gpu_batch) {
    size_t batch_row_begin = gpu_batch * gpu_batch_nrows;
    size_t batch_row_end = std::min((gpu_batch + 1) * gpu_batch_nrows, row_batch.Size());
    size_t batch_nrows = batch_row_end - batch_row_begin;

    const auto ent_cnt_begin = offset_vec[batch_row_begin];
    const auto ent_cnt_end = offset_vec[batch_row_end];

    /*! \brief row offset in SparsePage (the input data). */
    using OffT = typename std::remove_reference_t<decltype(offset_vec)>::value_type;
    dh::DeviceUVector<OffT> row_ptrs(batch_nrows + 1);
    auto size =
        std::distance(offset_vec.data() + batch_row_begin, offset_vec.data() + batch_row_end + 1);
    dh::safe_cuda(cudaMemcpyAsync(row_ptrs.data(), offset_vec.data() + batch_row_begin,
                                  size * sizeof(OffT), cudaMemcpyDefault,
                                  ctx->CUDACtx()->Stream()));

    // number of entries in this batch.
    size_t n_entries = ent_cnt_end - ent_cnt_begin;
    dh::DeviceUVector<Entry> entries_d(n_entries);
    // copy data entries to device.
    if (row_batch.data.DeviceCanRead()) {
      auto const& d_data = row_batch.data.ConstDeviceSpan();
      dh::safe_cuda(cudaMemcpyAsync(entries_d.data(), d_data.data() + ent_cnt_begin,
                                    n_entries * sizeof(Entry), cudaMemcpyDefault,
                                    ctx->CUDACtx()->Stream()));
    } else {
      const std::vector<Entry>& data_vec = row_batch.data.ConstHostVector();
      dh::safe_cuda(cudaMemcpyAsync(entries_d.data(), data_vec.data() + ent_cnt_begin,
                                    n_entries * sizeof(Entry), cudaMemcpyDefault,
                                    ctx->CUDACtx()->Stream()));
    }

    const dim3 block3(32, 8, 1);  // 256 threads
    const dim3 grid3(common::DivRoundUp(batch_nrows, block3.x),
                     common::DivRoundUp(this->info.row_stride, block3.y), 1);
    auto device_accessor = this->GetDeviceAccessor(ctx);
    auto launcher = [&](auto kernel) {
      dh::LaunchKernel{grid3, block3, 0, ctx->CUDACtx()->Stream()}(  // NOLINT
          kernel, writer, gidx_buffer_data, row_ptrs.data(), entries_d.data(),
          device_accessor.gidx_fvalue_map.data(), device_accessor.feature_segments, feature_types,
          batch_row_begin, batch_nrows, this->info.row_stride, null_gidx_value);
    };
    if (this->IsDense()) {
      launcher(CompressBinEllpackKernel<true, true>);
    } else {
      if (this->IsDenseCompressed()) {
        launcher(CompressBinEllpackKernel<false, true>);
      } else {
        launcher(CompressBinEllpackKernel<false, false>);
      }
    }
  }
  this->monitor_.Stop(__func__);
}

// Return the number of rows contained in this page.
[[nodiscard]] bst_idx_t EllpackPageImpl::Size() const { return n_rows; }

std::size_t EllpackPageImpl::MemCostBytes() const {
  return this->gidx_buffer.size_bytes() + sizeof(this->is_dense) + sizeof(this->n_rows) +
         sizeof(this->base_rowid) + sizeof(this->info);
}

EllpackDeviceAccessor EllpackPageImpl::GetDeviceAccessor(
    Context const* ctx, common::Span<FeatureType const> feature_types) const {
  auto null = this->IsDense() ? this->NumSymbols() : this->NumSymbols() - 1;
  return {ctx,
          this->cuts_,
          this->info.row_stride,
          this->base_rowid,
          this->n_rows,
          common::CompressedIterator<uint32_t>{gidx_buffer.data(), this->NumSymbols()},
          null,
          feature_types};
}

EllpackDeviceAccessor EllpackPageImpl::GetHostAccessor(
    Context const* ctx, std::vector<common::CompressedByteT>* h_gidx_buffer,
    common::Span<FeatureType const> feature_types) const {
  h_gidx_buffer->resize(gidx_buffer.size());
  CHECK_EQ(h_gidx_buffer->size(), gidx_buffer.size());
  CHECK_NE(gidx_buffer.size(), 0);
  dh::safe_cuda(cudaMemcpyAsync(h_gidx_buffer->data(), gidx_buffer.data(), gidx_buffer.size_bytes(),
                                cudaMemcpyDefault, ctx->CUDACtx()->Stream()));
  Context cpu_ctx;
  auto null = this->IsDense() ? this->NumSymbols() : this->NumSymbols() - 1;
  return {ctx->IsCPU() ? ctx : &cpu_ctx,
          this->cuts_,
          this->info.row_stride,
          this->base_rowid,
          this->n_rows,
          common::CompressedIterator<uint32_t>{h_gidx_buffer->data(), this->NumSymbols()},
          null,
          feature_types};
}

[[nodiscard]] bst_idx_t EllpackPageImpl::NumNonMissing(
    Context const* ctx, common::Span<FeatureType const> feature_types) const {
  if (this->IsDense()) {
    return this->n_rows * this->info.row_stride;
  }
  auto d_acc = this->GetDeviceAccessor(ctx, feature_types);
  using T = typename decltype(d_acc.gidx_iter)::value_type;
  auto it = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0ull),
      [=] XGBOOST_DEVICE(std::size_t i) { return d_acc.gidx_iter[i]; });
  return thrust::count_if(ctx->CUDACtx()->CTP(), it, it + d_acc.row_stride * d_acc.n_rows,
                          [=] XGBOOST_DEVICE(T gidx) -> bool { return gidx != d_acc.NullValue(); });
}
}  // namespace xgboost
