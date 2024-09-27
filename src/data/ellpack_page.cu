/**
 * Copyright 2019-2024, XGBoost contributors
 */
#include <cuda/functional>  // for proclaim_return_type
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

#include <algorithm>        // for copy
#include <utility>          // for move
#include <vector>           // for vector

#include "../common/algorithm.cuh"  // for InclusiveScan
#include "../common/categorical.h"
#include "../common/cuda_context.cuh"
#include "../common/cuda_rt_utils.h"        // for SetDevice
#include "../common/hist_util.cuh"          // for HistogramCuts
#include "../common/ref_resource_view.cuh"  // for MakeFixedVecWithCudaMalloc
#include "../common/transform_iterator.h"   // for MakeIndexTransformIter
#include "device_adapter.cuh"               // for NoInfInData
#include "ellpack_page.cuh"
#include "ellpack_page.h"
#include "gradient_index.h"
#include "xgboost/data.h"

namespace xgboost {

EllpackPage::EllpackPage() : impl_{new EllpackPageImpl()} {}

EllpackPage::EllpackPage(Context const* ctx, DMatrix* dmat, const BatchParam& param)
    : impl_{new EllpackPageImpl{ctx, dmat, param}} {}

EllpackPage::~EllpackPage() = default;

EllpackPage::EllpackPage(EllpackPage&& that) { std::swap(impl_, that.impl_); }

[[nodiscard]] bst_idx_t EllpackPage::Size() const { return impl_->Size(); }

void EllpackPage::SetBaseRowId(std::size_t row_id) { impl_->SetBaseRowId(row_id); }

[[nodiscard]] common::HistogramCuts const& EllpackPage::Cuts() const {
  CHECK(impl_);
  return impl_->Cuts();
}

[[nodiscard]] bst_idx_t EllpackPage::BaseRowId() const { return this->Impl()->base_rowid; }
[[nodiscard]] bool EllpackPage::IsDense() const { return this->Impl()->IsDense(); }

// Bin each input data entry, store the bin indices in compressed form.
template <bool kIsDense>
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
  size_t irow = threadIdx.x + blockIdx.x * blockDim.x;
  int ifeature = threadIdx.y + blockIdx.y * blockDim.y;
  if (irow >= n_rows || ifeature >= row_stride) {
    return;
  }
  int row_length = static_cast<int>(row_ptrs[irow + 1] - row_ptrs[irow]);
  std::uint32_t bin = null_gidx_value;
  if (ifeature < row_length) {
    Entry entry = entries[row_ptrs[irow] - row_ptrs[0] + ifeature];
    int feature = entry.index;
    float fvalue = entry.fvalue;
    // {feature_cuts, ncuts} forms the array of cuts of `feature'.
    const float* feature_cuts = &cuts[cut_ptrs[feature]];
    int ncuts = cut_ptrs[feature + 1] - cut_ptrs[feature];
    bool is_cat = common::IsCat(feature_types, ifeature);
    // Assigning the bin in current entry.
    // S.t.: fvalue < feature_cuts[bin]
    if (is_cat) {
      auto it =
          dh::MakeTransformIterator<int>(feature_cuts, [](float v) { return common::AsCat(v); });
      bin = thrust::lower_bound(thrust::seq, it, it + ncuts, common::AsCat(fvalue)) - it;
    } else {
      bin = thrust::upper_bound(thrust::seq, feature_cuts, feature_cuts + ncuts, fvalue) -
            feature_cuts;
    }

    if (bin >= ncuts) {
      bin = ncuts - 1;
    }
    // Add the number of bins in previous features.
    if (!kIsDense) {
      bin += cut_ptrs[feature];
    }
  }
  // Write to gidx buffer.
  wr.AtomicWriteSymbol(buffer, bin, (irow + base_row) * row_stride + ifeature);
}

namespace {
// Calculate the number of symbols for the compressed ellpack. Similar to what the CPU
// implementation does, we compress the dense data by subtracting the bin values with the
// starting bin of its feature.
[[nodiscard]] std::size_t CalcNumSymbols(Context const* ctx, bool is_dense,
                                         std::shared_ptr<common::HistogramCuts const> cuts) {
  // Cut values can be empty when the input data is empty.
  if (!is_dense || cuts->cut_values_.Empty()) {
    // Return the total number of symbols (total number of bins plus 1 for not found)
    return cuts->cut_values_.Size() + 1;
  }

  cuts->cut_ptrs_.SetDevice(ctx->Device());
  common::Span<std::uint32_t const> dptrs = cuts->cut_ptrs_.ConstDeviceSpan();
  auto cuctx = ctx->CUDACtx();
  using PtrT = typename decltype(dptrs)::value_type;
  auto it = dh::MakeTransformIterator<PtrT>(
      thrust::make_counting_iterator(1ul),
      [=] XGBOOST_DEVICE(std::size_t i) { return dptrs[i] - dptrs[i - 1]; });
  CHECK_GE(dptrs.size(), 2);
  auto max_it = thrust::max_element(cuctx->CTP(), it, it + dptrs.size() - 1);
  dh::CachingDeviceUVector<PtrT> max_element(1);
  auto d_me = max_element.data();
  dh::LaunchN(1, cuctx->Stream(), [=] XGBOOST_DEVICE(std::size_t i) { d_me[i] = *max_it; });
  PtrT h_me{0};
  dh::safe_cuda(
      cudaMemcpyAsync(&h_me, d_me, sizeof(PtrT), cudaMemcpyDeviceToHost, cuctx->Stream()));
  cuctx->Stream().Sync();
  // No missing, hence no null value, hence no + 1 symbol.
  // FIXME(jiamingy): When we extend this to use a sparsity threshold, +1 is needed back.
  return h_me;
}
}  // namespace

// Construct an ELLPACK matrix with the given number of empty rows.
EllpackPageImpl::EllpackPageImpl(Context const* ctx,
                                 std::shared_ptr<common::HistogramCuts const> cuts, bool is_dense,
                                 bst_idx_t row_stride, bst_idx_t n_rows)
    : is_dense{is_dense},
      cuts_{std::move(cuts)},
      row_stride{row_stride},
      n_rows{n_rows},
      n_symbols_{CalcNumSymbols(ctx, this->is_dense, this->cuts_)} {
  monitor_.Init("ellpack_page");
  curt::SetDevice(ctx->Ordinal());

  this->InitCompressedData(ctx);
}

EllpackPageImpl::EllpackPageImpl(Context const* ctx,
                                 std::shared_ptr<common::HistogramCuts const> cuts,
                                 const SparsePage& page, bool is_dense, size_t row_stride,
                                 common::Span<FeatureType const> feature_types)
    : cuts_{std::move(cuts)},
      is_dense{is_dense},
      n_rows{page.Size()},
      row_stride{row_stride},
      n_symbols_{CalcNumSymbols(ctx, this->is_dense, this->cuts_)} {
  monitor_.Init("ellpack_page");
  curt::SetDevice(ctx->Ordinal());

  this->InitCompressedData(ctx);
  this->CreateHistIndices(ctx, page, feature_types);
}

// Construct an ELLPACK matrix in memory.
EllpackPageImpl::EllpackPageImpl(Context const* ctx, DMatrix* p_fmat, const BatchParam& param)
    : is_dense{p_fmat->IsDense()},
      n_rows{p_fmat->Info().num_row_},
      row_stride{GetRowStride(p_fmat)},
      // Create the quantile sketches for the dmatrix and initialize HistogramCuts.
      cuts_{param.hess.empty()
                ? std::make_shared<common::HistogramCuts>(
                      common::DeviceSketch(ctx, p_fmat, param.max_bin))
                : std::make_shared<common::HistogramCuts>(
                      common::DeviceSketchWithHessian(ctx, p_fmat, param.max_bin, param.hess))},
      n_symbols_{CalcNumSymbols(ctx, this->is_dense, this->cuts_)} {
  monitor_.Init("ellpack_page");
  curt::SetDevice(ctx->Ordinal());

  this->InitCompressedData(ctx);

  p_fmat->Info().feature_types.SetDevice(ctx->Device());
  auto ft = p_fmat->Info().feature_types.ConstDeviceSpan();
  monitor_.Start("BinningCompression");
  CHECK(p_fmat->SingleColBlock());
  for (auto const& page : p_fmat->GetBatches<SparsePage>()) {
    this->CreateHistIndices(ctx, page, ft);
  }
  monitor_.Stop("BinningCompression");
}

template <typename AdapterBatchT, bool kIsDense>
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

  using Tuple = thrust::tuple<size_t, size_t, size_t>;
  __device__ size_t operator()(Tuple out) {
    auto e = batch.GetElement(thrust::get<2>(out));
    if (is_valid(e)) {
      // -1 because the scan is inclusive
      size_t output_position = accessor.row_stride * e.row_idx + thrust::get<1>(out) - 1;
      uint32_t bin_idx = 0;
      if (common::IsCat(feature_types, e.column_idx)) {
        bin_idx = accessor.SearchBin<true>(e.value, e.column_idx);
      } else {
        bin_idx = accessor.SearchBin<false>(e.value, e.column_idx);
      }
      if (kIsDense) {
        bin_idx -= accessor.feature_segments[e.column_idx];
      }
      writer.AtomicWriteSymbol(d_buffer, bin_idx, output_position);
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
template <bool kIsDense, typename AdapterBatchT>
void CopyDataToEllpack(Context const* ctx, const AdapterBatchT& batch,
                       common::Span<FeatureType const> feature_types, EllpackPageImpl* dst,
                       float missing) {
  // Some witchcraft happens here
  // The goal is to copy valid elements out of the input to an ELLPACK matrix
  // with a given row stride, using no extra working memory Standard stream
  // compaction needs to be modified to do this, so we manually define a
  // segmented stream compaction via operators on an inclusive scan. The output
  // of this inclusive scan is fed to a custom function which works out the
  // correct output position
  auto counting = thrust::make_counting_iterator(0llu);
  data::IsValidFunctor is_valid(missing);
  bool valid = data::NoInfInData(batch, is_valid);
  CHECK(valid) << error::InfInData();

  auto key_iter = dh::MakeTransformIterator<size_t>(
      counting,
      [=] __device__(size_t idx) {
        return batch.GetElement(idx).row_idx;
      });
  auto value_iter = dh::MakeTransformIterator<size_t>(
      counting,
      [=] __device__(size_t idx) -> size_t {
        return is_valid(batch.GetElement(idx));
      });

  auto key_value_index_iter =
      thrust::make_zip_iterator(thrust::make_tuple(key_iter, value_iter, counting));

  // Tuple[0] = The row index of the input, used as a key to define segments
  // Tuple[1] = Scanned flags of valid elements for each row
  // Tuple[2] = The index in the input data
  using Tuple = thrust::tuple<bst_idx_t, bst_idx_t, bst_idx_t>;

  auto device_accessor = dst->GetDeviceAccessor(ctx);
  auto n_symbols = dst->NumSymbols();

  common::CompressedBufferWriter writer{n_symbols};
  auto d_compressed_buffer = dst->gidx_buffer.data();

  // We redirect the scan output into this functor to do the actual writing
  dh::TypedDiscard<Tuple> discard;
  WriteCompressedEllpackFunctor<AdapterBatchT, kIsDense> functor{
      d_compressed_buffer, writer, batch, device_accessor, feature_types, is_valid};
  thrust::transform_output_iterator<decltype(functor), decltype(discard)> out(discard, functor);

  common::InclusiveScan(ctx, key_value_index_iter, out, TupleScanOp<Tuple>{}, batch.Size());
}

void WriteNullValues(Context const* ctx, EllpackPageImpl* dst,
                     common::Span<size_t const> row_counts) {
  // Write the null values
  auto device_accessor = dst->GetDeviceAccessor(ctx);
  common::CompressedBufferWriter writer(dst->NumSymbols());
  auto d_compressed_buffer = dst->gidx_buffer.data();
  auto row_stride = dst->row_stride;
  dh::LaunchN(row_stride * dst->n_rows, ctx->CUDACtx()->Stream(), [=] __device__(bst_idx_t idx) {
    // For some reason this variable got captured as const
    auto writer_non_const = writer;
    size_t row_idx = idx / row_stride;
    size_t row_offset = idx % row_stride;
    if (row_offset >= row_counts[row_idx]) {
      writer_non_const.AtomicWriteSymbol(d_compressed_buffer, device_accessor.NullValue(), idx);
    }
  });
}

template <typename AdapterBatch>
EllpackPageImpl::EllpackPageImpl(Context const* ctx, AdapterBatch batch, float missing,
                                 bool is_dense, common::Span<size_t const> row_counts_span,
                                 common::Span<FeatureType const> feature_types, size_t row_stride,
                                 bst_idx_t n_rows,
                                 std::shared_ptr<common::HistogramCuts const> cuts)
    : EllpackPageImpl{ctx, cuts, is_dense, row_stride, n_rows} {
  curt::SetDevice(ctx->Ordinal());

  if (this->IsDense()) {
    CopyDataToEllpack<true>(ctx, batch, feature_types, this, missing);
  } else {
    CopyDataToEllpack<false>(ctx, batch, feature_types, this, missing);
  }

  WriteNullValues(ctx, this, row_counts_span);
}

#define ELLPACK_BATCH_SPECIALIZE(__BATCH_T)                                                      \
  template EllpackPageImpl::EllpackPageImpl(                                                     \
      Context const* ctx, __BATCH_T batch, float missing, bool is_dense,                         \
      common::Span<size_t const> row_counts_span, common::Span<FeatureType const> feature_types, \
      size_t row_stride, size_t n_rows, std::shared_ptr<common::HistogramCuts const> cuts);

ELLPACK_BATCH_SPECIALIZE(data::CudfAdapterBatch)
ELLPACK_BATCH_SPECIALIZE(data::CupyAdapterBatch)

namespace {
void CopyGHistToEllpack(Context const* ctx, GHistIndexMatrix const& page,
                        common::Span<bst_idx_t const> d_row_ptr, bst_idx_t row_stride,
                        bst_bin_t null, bst_idx_t n_symbols,
                        common::CompressedByteT* d_compressed_buffer) {
  dh::device_vector<uint8_t> data(page.index.begin(), page.index.end());
  auto d_data = dh::ToSpan(data);

  // GPU employs the same dense compression as CPU, no need to handle page.index.Offset()
  auto bin_type = page.index.GetBinTypeSize();
  common::CompressedBufferWriter writer{n_symbols};

  auto cuctx = ctx->CUDACtx();
  dh::LaunchN(row_stride * page.Size(), cuctx->Stream(), [=] __device__(bst_idx_t idx) mutable {
    auto ridx = idx / row_stride;
    auto ifeature = idx % row_stride;

    auto r_begin = d_row_ptr[ridx];
    auto r_end = d_row_ptr[ridx + 1];
    auto r_size = r_end - r_begin;

    if (ifeature >= r_size) {
      writer.AtomicWriteSymbol(d_compressed_buffer, null, idx);
      return;
    }

    common::cuda::DispatchBinType(bin_type, [&](auto t) {
      using T = decltype(t);
      auto ptr = reinterpret_cast<T const*>(d_data.data());
      auto bin_idx = ptr[r_begin + ifeature];
      writer.AtomicWriteSymbol(d_compressed_buffer, bin_idx, idx);
    });
  });
}
}  // anonymous namespace

EllpackPageImpl::EllpackPageImpl(Context const* ctx, GHistIndexMatrix const& page,
                                 common::Span<FeatureType const> ft)
    : is_dense{page.IsDense()},
      row_stride{[&] {
        auto it = common::MakeIndexTransformIter(
            [&](bst_idx_t i) { return page.row_ptr[i + 1] - page.row_ptr[i]; });
        return *std::max_element(it, it + page.Size());
      }()},
      base_rowid{page.base_rowid},
      n_rows{page.Size()},
      cuts_{std::make_shared<common::HistogramCuts>(page.cut)},
      n_symbols_{CalcNumSymbols(ctx, page.IsDense(), cuts_)} {
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
  CopyGHistToEllpack(ctx, page, d_row_ptr, row_stride, accessor.NullValue(), this->NumSymbols(),
                     d_compressed_buffer);
  this->monitor_.Stop("CopyGHistToEllpack");
}

EllpackPageImpl::~EllpackPageImpl() noexcept(false) {
  // Sync the stream to make sure all running CUDA kernels finish before deallocation.
  dh::DefaultStream().Sync();
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

  __device__ void operator()(size_t element_id) {
    cbw.AtomicWriteSymbol(dst_data_d, src_iterator_d[element_id], element_id + offset);
  }
};

// Copy the data from the given EllpackPage to the current page.
bst_idx_t EllpackPageImpl::Copy(Context const* ctx, EllpackPageImpl const* page, bst_idx_t offset) {
  monitor_.Start(__func__);
  bst_idx_t num_elements = page->n_rows * page->row_stride;
  CHECK_EQ(this->row_stride, page->row_stride);
  CHECK_EQ(NumSymbols(), page->NumSymbols());
  CHECK_GE(this->n_rows * this->row_stride, offset + num_elements);
  if (page == this) {
    LOG(FATAL) << "Concatenating the same Ellpack.";
    return this->n_rows * this->row_stride;
  }
  dh::LaunchN(num_elements, ctx->CUDACtx()->Stream(), CopyPage{this, page, offset});
  monitor_.Stop(__func__);
  return num_elements;
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
        row_stride{src->row_stride} {}

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
  CHECK_EQ(row_stride, page->row_stride);
  CHECK_EQ(NumSymbols(), page->NumSymbols());
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
  std::size_t compressed_size_bytes =
      common::CompressedBufferWriter::CalculateBufferSize(row_stride * n_rows, num_symbols);
  auto init = static_cast<common::CompressedByteT>(0);
  gidx_buffer = common::MakeFixedVecWithCudaMalloc(ctx, compressed_size_bytes, init);
  monitor_.Stop(__func__);
}

// Compress a CSR page into ELLPACK.
void EllpackPageImpl::CreateHistIndices(Context const* ctx,
                                        const SparsePage& row_batch,
                                        common::Span<FeatureType const> feature_types) {
  if (row_batch.Size() == 0) {
    return;
  }
  auto null_gidx_value = this->GetDeviceAccessor(ctx, feature_types).NullValue();

  auto const& offset_vec = row_batch.offset.ConstHostVector();

  // bin and compress entries in batches of rows
  size_t gpu_batch_nrows =
      std::min(dh::TotalMemory(ctx->Ordinal()) / (16 * row_stride * sizeof(Entry)),
               static_cast<size_t>(row_batch.Size()));

  size_t gpu_nbatches = common::DivRoundUp(row_batch.Size(), gpu_batch_nrows);

  for (size_t gpu_batch = 0; gpu_batch < gpu_nbatches; ++gpu_batch) {
    size_t batch_row_begin = gpu_batch * gpu_batch_nrows;
    size_t batch_row_end =
        std::min((gpu_batch + 1) * gpu_batch_nrows, row_batch.Size());
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
                     common::DivRoundUp(row_stride, block3.y), 1);
    auto device_accessor = this->GetDeviceAccessor(ctx);
    auto launcher = [&](auto kernel) {
      dh::LaunchKernel{grid3, block3, 0, ctx->CUDACtx()->Stream()}(  // NOLINT
          kernel, common::CompressedBufferWriter(this->NumSymbols()), gidx_buffer.data(),
          row_ptrs.data(), entries_d.data(), device_accessor.gidx_fvalue_map.data(),
          device_accessor.feature_segments.data(), feature_types, batch_row_begin, batch_nrows,
          row_stride, null_gidx_value);
    };
    if (this->IsDense()) {
      launcher(CompressBinEllpackKernel<true>);
    } else {
      launcher(CompressBinEllpackKernel<false>);
    }
  }
}

// Return the number of rows contained in this page.
[[nodiscard]] bst_idx_t EllpackPageImpl::Size() const { return n_rows; }

std::size_t EllpackPageImpl::MemCostBytes() const {
  return this->gidx_buffer.size_bytes() + sizeof(this->n_rows) + sizeof(this->is_dense) +
         sizeof(this->row_stride) + sizeof(this->base_rowid) + sizeof(this->n_symbols_);
}

EllpackDeviceAccessor EllpackPageImpl::GetDeviceAccessor(
    Context const* ctx, common::Span<FeatureType const> feature_types) const {
  return {ctx,
          cuts_,
          is_dense,
          row_stride,
          base_rowid,
          n_rows,
          common::CompressedIterator<uint32_t>(gidx_buffer.data(), this->NumSymbols()),
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
  return {ctx->IsCPU() ? ctx : &cpu_ctx,
          cuts_,
          is_dense,
          row_stride,
          base_rowid,
          n_rows,
          common::CompressedIterator<uint32_t>(h_gidx_buffer->data(), this->NumSymbols()),
          feature_types};
}

[[nodiscard]] bst_idx_t EllpackPageImpl::NumNonMissing(
    Context const* ctx, common::Span<FeatureType const> feature_types) const {
  auto d_acc = this->GetDeviceAccessor(ctx, feature_types);
  using T = typename decltype(d_acc.gidx_iter)::value_type;
  auto it = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0ull),
      cuda::proclaim_return_type<T>([=] __device__(std::size_t i) { return d_acc.gidx_iter[i]; }));
  auto nnz = thrust::count_if(ctx->CUDACtx()->CTP(), it, it + d_acc.row_stride * d_acc.n_rows,
                              cuda::proclaim_return_type<bool>(
                                  [=] __device__(T gidx) { return gidx != d_acc.NullValue(); }));
  return nnz;
}
}  // namespace xgboost
