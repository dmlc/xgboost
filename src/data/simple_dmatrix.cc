/**
 * Copyright 2014~2023 by XGBoost Contributors
 * \file simple_dmatrix.cc
 * \brief the input data structure for gradient boosting
 * \author Tianqi Chen
 */
#include "simple_dmatrix.h"

#include <algorithm>
#include <limits>
#include <type_traits>
#include <vector>

#include "../common/error_msg.h"  // for InconsistentMaxBin
#include "../common/random.h"
#include "../common/threading_utils.h"
#include "./simple_batch_iterator.h"
#include "adapter.h"
#include "batch_utils.h"  // for CheckEmpty, RegenGHist
#include "gradient_index.h"
#include "xgboost/c_api.h"
#include "xgboost/data.h"

namespace xgboost {
namespace data {
MetaInfo& SimpleDMatrix::Info() { return info_; }

const MetaInfo& SimpleDMatrix::Info() const { return info_; }

DMatrix* SimpleDMatrix::Slice(common::Span<int32_t const> ridxs) {
  auto out = new SimpleDMatrix;
  SparsePage& out_page = *out->sparse_page_;
  for (auto const& page : this->GetBatches<SparsePage>()) {
    auto batch = page.GetView();
    auto& h_data = out_page.data.HostVector();
    auto& h_offset = out_page.offset.HostVector();
    size_t rptr{0};
    for (auto ridx : ridxs) {
      auto inst = batch[ridx];
      rptr += inst.size();
      std::copy(inst.begin(), inst.end(), std::back_inserter(h_data));
      h_offset.emplace_back(rptr);
    }
    out->Info() = this->Info().Slice(ridxs);
    out->Info().num_nonzero_ = h_offset.back();
  }
  out->fmat_ctx_ = this->fmat_ctx_;
  return out;
}

DMatrix* SimpleDMatrix::SliceCol(int num_slices, int slice_id) {
  auto out = new SimpleDMatrix;
  SparsePage& out_page = *out->sparse_page_;
  auto const slice_size = info_.num_col_ / num_slices;
  auto const slice_start = slice_size * slice_id;
  auto const slice_end = (slice_id == num_slices - 1) ? info_.num_col_ : slice_start + slice_size;
  for (auto const& page : this->GetBatches<SparsePage>()) {
    auto batch = page.GetView();
    auto& h_data = out_page.data.HostVector();
    auto& h_offset = out_page.offset.HostVector();
    size_t rptr{0};
    for (bst_row_t i = 0; i < this->Info().num_row_; i++) {
      auto inst = batch[i];
      auto prev_size = h_data.size();
      std::copy_if(inst.begin(), inst.end(), std::back_inserter(h_data),
                   [&](Entry e) { return e.index >= slice_start && e.index < slice_end; });
      rptr += h_data.size() - prev_size;
      h_offset.emplace_back(rptr);
    }
    out->Info() = this->Info().Copy();
    out->Info().num_nonzero_ = h_offset.back();
  }
  out->Info().data_split_mode = DataSplitMode::kCol;
  return out;
}

void SimpleDMatrix::ReindexFeatures(Context const* ctx) {
  if (info_.IsVerticalFederated()) {
    std::vector<uint64_t> buffer(collective::GetWorldSize());
    buffer[collective::GetRank()] = info_.num_col_;
    collective::Allgather(buffer.data(), buffer.size() * sizeof(uint64_t));
    auto offset = std::accumulate(buffer.cbegin(), buffer.cbegin() + collective::GetRank(), 0);
    if (offset == 0) {
      return;
    }
    sparse_page_->Reindex(offset, ctx->Threads());
  }
}

BatchSet<SparsePage> SimpleDMatrix::GetRowBatches() {
  // since csr is the default data structure so `source_` is always available.
  auto begin_iter =
      BatchIterator<SparsePage>(new SimpleBatchIteratorImpl<SparsePage>(sparse_page_));
  return BatchSet<SparsePage>(begin_iter);
}

BatchSet<CSCPage> SimpleDMatrix::GetColumnBatches(Context const* ctx) {
  // column page doesn't exist, generate it
  if (!column_page_) {
    column_page_.reset(new CSCPage(sparse_page_->GetTranspose(info_.num_col_, ctx->Threads())));
  }
  auto begin_iter = BatchIterator<CSCPage>(new SimpleBatchIteratorImpl<CSCPage>(column_page_));
  return BatchSet<CSCPage>(begin_iter);
}

BatchSet<SortedCSCPage> SimpleDMatrix::GetSortedColumnBatches(Context const* ctx) {
  // Sorted column page doesn't exist, generate it
  if (!sorted_column_page_) {
    sorted_column_page_.reset(
        new SortedCSCPage(sparse_page_->GetTranspose(info_.num_col_, ctx->Threads())));
    sorted_column_page_->SortRows(ctx->Threads());
  }
  auto begin_iter =
      BatchIterator<SortedCSCPage>(new SimpleBatchIteratorImpl<SortedCSCPage>(sorted_column_page_));
  return BatchSet<SortedCSCPage>(begin_iter);
}

BatchSet<EllpackPage> SimpleDMatrix::GetEllpackBatches(Context const* ctx,
                                                       const BatchParam& param) {
  detail::CheckEmpty(batch_param_, param);
  if (ellpack_page_ && param.Initialized() && param.forbid_regen) {
    if (detail::RegenGHist(batch_param_, param)) {
      CHECK_EQ(batch_param_.max_bin, param.max_bin) << error::InconsistentMaxBin();
    }
    CHECK(!detail::RegenGHist(batch_param_, param));
  }
  if (!ellpack_page_ || detail::RegenGHist(batch_param_, param)) {
    // ELLPACK page doesn't exist, generate it
    LOG(INFO) << "Generating new Ellpack page.";
    // These places can ask for a ellpack page:
    // - GPU hist: the ctx must be on CUDA.
    // - IterativeDMatrix::InitFromCUDA: The ctx must be on CUDA.
    // - IterativeDMatrix::InitFromCPU: It asks for ellpack only if it exists. It should
    //   not regen, otherwise it indicates a mismatched parameter like max_bin.
    CHECK_GE(param.max_bin, 2);
    if (ctx->IsCUDA()) {
      // The context passed in is on GPU, we pick it first since we prioritize the context
      // in Booster.
      ellpack_page_.reset(new EllpackPage(ctx, this, param));
    } else if (fmat_ctx_.IsCUDA()) {
      // DMatrix was initialized on GPU, we use the context from initialization.
      ellpack_page_.reset(new EllpackPage(&fmat_ctx_, this, param));
    } else {
      // Mismatched parameter, user set a new max_bin during training.
      auto cuda_ctx = ctx->MakeCUDA();
      ellpack_page_.reset(new EllpackPage(&cuda_ctx, this, param));
    }

    batch_param_ = param.MakeCache();
  }
  auto begin_iter =
      BatchIterator<EllpackPage>(new SimpleBatchIteratorImpl<EllpackPage>(ellpack_page_));
  return BatchSet<EllpackPage>(begin_iter);
}

BatchSet<GHistIndexMatrix> SimpleDMatrix::GetGradientIndex(Context const* ctx,
                                                           const BatchParam& param) {
  detail::CheckEmpty(batch_param_, param);
  // Check whether we can regenerate the gradient index. This is to keep the consistency
  // between evaluation data and training data.
  if (gradient_index_ && param.Initialized() && param.forbid_regen) {
    if (detail::RegenGHist(batch_param_, param)) {
      CHECK_EQ(batch_param_.max_bin, param.max_bin) << error::InconsistentMaxBin();
    }
    CHECK(!detail::RegenGHist(batch_param_, param)) << "Inconsistent sparse threshold.";
  }
  if (!gradient_index_ || detail::RegenGHist(batch_param_, param)) {
    // GIDX page doesn't exist, generate it
    LOG(DEBUG) << "Generating new Gradient Index.";
    // These places can ask for a CSR gidx:
    // - CPU Hist: the ctx must be on CPU.
    // - IterativeDMatrix::InitFromCPU: The ctx must be on CPU.
    // - IterativeDMatrix::InitFromCUDA: It asks for gidx only if it exists. It should not
    //   regen, otherwise it indicates a mismatched parameter like max_bin.
    CHECK_GE(param.max_bin, 2);
    // Used only by approx.
    auto sorted_sketch = param.regen;
    if (ctx->IsCPU()) {
      // The context passed in is on CPU, we pick it first since we prioritize the context
      // in Booster.
      gradient_index_.reset(new GHistIndexMatrix{ctx, this, param.max_bin, param.sparse_thresh,
                                                 sorted_sketch, param.hess});
    } else if (fmat_ctx_.IsCPU()) {
      // DMatrix was initialized on CPU, we use the context from initialization.
      gradient_index_.reset(new GHistIndexMatrix{&fmat_ctx_, this, param.max_bin,
                                                 param.sparse_thresh, sorted_sketch, param.hess});
    } else {
      // Mismatched parameter, user set a new max_bin during training.
      auto cpu_ctx = ctx->MakeCPU();
      gradient_index_.reset(new GHistIndexMatrix{&cpu_ctx, this, param.max_bin, param.sparse_thresh,
                                                 sorted_sketch, param.hess});
    }

    batch_param_ = param.MakeCache();
    CHECK_EQ(batch_param_.hess.data(), param.hess.data());
  }
  auto begin_iter = BatchIterator<GHistIndexMatrix>(
      new SimpleBatchIteratorImpl<GHistIndexMatrix>(gradient_index_));
  return BatchSet<GHistIndexMatrix>(begin_iter);
}

BatchSet<ExtSparsePage> SimpleDMatrix::GetExtBatches(Context const*, BatchParam const&) {
  auto casted = std::make_shared<ExtSparsePage>(sparse_page_);
  CHECK(casted);
  auto begin_iter =
      BatchIterator<ExtSparsePage>(new SimpleBatchIteratorImpl<ExtSparsePage>(casted));
  return BatchSet<ExtSparsePage>(begin_iter);
}

template <typename AdapterT>
SimpleDMatrix::SimpleDMatrix(AdapterT* adapter, float missing, int nthread,
                             DataSplitMode data_split_mode) {
  Context ctx;
  ctx.Init(Args{{"nthread", std::to_string(nthread)}});

  std::vector<uint64_t> qids;
  uint64_t default_max = std::numeric_limits<uint64_t>::max();
  uint64_t last_group_id = default_max;
  bst_uint group_size = 0;
  auto& offset_vec = sparse_page_->offset.HostVector();
  auto& data_vec = sparse_page_->data.HostVector();
  uint64_t inferred_num_columns = 0;
  uint64_t total_batch_size = 0;
  // batch_size is either number of rows or cols, depending on data layout

  adapter->BeforeFirst();
  // Iterate over batches of input data
  while (adapter->Next()) {
    auto& batch = adapter->Value();
    auto batch_max_columns = sparse_page_->Push(batch, missing, ctx.Threads());
    inferred_num_columns = std::max(batch_max_columns, inferred_num_columns);
    total_batch_size += batch.Size();
    // Append meta information if available
    if (batch.Labels() != nullptr) {
      info_.labels.ModifyInplace([&](auto* data, common::Span<size_t, 2> shape) {
        shape[1] = 1;
        auto& labels = data->HostVector();
        labels.insert(labels.end(), batch.Labels(), batch.Labels() + batch.Size());
        shape[0] += batch.Size();
      });
    }
    if (batch.Weights() != nullptr) {
      auto& weights = info_.weights_.HostVector();
      weights.insert(weights.end(), batch.Weights(), batch.Weights() + batch.Size());
    }
    if (batch.BaseMargin() != nullptr) {
      info_.base_margin_ = decltype(info_.base_margin_){
          batch.BaseMargin(), batch.BaseMargin() + batch.Size(), {batch.Size()}, Context::kCpuId};
    }
    if (batch.Qid() != nullptr) {
      qids.insert(qids.end(), batch.Qid(), batch.Qid() + batch.Size());
      // get group
      for (size_t i = 0; i < batch.Size(); ++i) {
        const uint64_t cur_group_id = batch.Qid()[i];
        if (last_group_id == default_max || last_group_id != cur_group_id) {
          info_.group_ptr_.push_back(group_size);
        }
        last_group_id = cur_group_id;
        ++group_size;
      }
    }
  }

  if (last_group_id != default_max) {
    if (group_size > info_.group_ptr_.back()) {
      info_.group_ptr_.push_back(group_size);
    }
  }

  // Deal with empty rows/columns if necessary
  if (adapter->NumColumns() == kAdapterUnknownSize) {
    info_.num_col_ = inferred_num_columns;
  } else {
    info_.num_col_ = adapter->NumColumns();
  }

  // Synchronise worker columns
  info_.data_split_mode = data_split_mode;
  ReindexFeatures(&ctx);
  info_.SynchronizeNumberOfColumns();

  if (adapter->NumRows() == kAdapterUnknownSize) {
    using IteratorAdapterT =
        IteratorAdapter<DataIterHandle, XGBCallbackDataIterNext, XGBoostBatchCSR>;
    // If AdapterT is either IteratorAdapter or FileAdapter type, use the total batch size to
    // determine the correct number of rows, as offset_vec may be too short
    if (std::is_same<AdapterT, IteratorAdapterT>::value ||
        std::is_same<AdapterT, FileAdapter>::value) {
      info_.num_row_ = total_batch_size;
      // Ensure offset_vec.size() - 1 == [number of rows]
      while (offset_vec.size() - 1 < total_batch_size) {
        offset_vec.emplace_back(offset_vec.back());
      }
    } else {
      CHECK((std::is_same<AdapterT, CSCAdapter>::value ||
             std::is_same<AdapterT, CSCArrayAdapter>::value))
          << "Expecting CSCAdapter";
      info_.num_row_ = offset_vec.size() - 1;
    }
  } else {
    if (offset_vec.empty()) {
      offset_vec.emplace_back(0);
    }
    while (offset_vec.size() - 1 < adapter->NumRows()) {
      offset_vec.emplace_back(offset_vec.back());
    }
    info_.num_row_ = adapter->NumRows();
  }
  info_.num_nonzero_ = data_vec.size();

  // Sort the index for row partitioners used by variuos tree methods.
  if (!sparse_page_->IsIndicesSorted(ctx.Threads())) {
    sparse_page_->SortIndices(ctx.Threads());
  }

  this->fmat_ctx_ = ctx;
}

SimpleDMatrix::SimpleDMatrix(dmlc::Stream* in_stream) {
  int tmagic;
  CHECK(in_stream->Read(&tmagic)) << "invalid input file format";
  CHECK_EQ(tmagic, kMagic) << "invalid format, magic number mismatch";
  info_.LoadBinary(in_stream);
  in_stream->Read(&sparse_page_->offset.HostVector());
  in_stream->Read(&sparse_page_->data.HostVector());
}

void SimpleDMatrix::SaveToLocalFile(const std::string& fname) {
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname.c_str(), "w"));
  int tmagic = kMagic;
  fo->Write(tmagic);
  info_.SaveBinary(fo.get());
  fo->Write(sparse_page_->offset.HostVector());
  fo->Write(sparse_page_->data.HostVector());
}

template SimpleDMatrix::SimpleDMatrix(DenseAdapter* adapter, float missing, int nthread,
                                      DataSplitMode data_split_mode);
template SimpleDMatrix::SimpleDMatrix(ArrayAdapter* adapter, float missing, int nthread,
                                      DataSplitMode data_split_mode);
template SimpleDMatrix::SimpleDMatrix(CSRAdapter* adapter, float missing, int nthread,
                                      DataSplitMode data_split_mode);
template SimpleDMatrix::SimpleDMatrix(CSRArrayAdapter* adapter, float missing, int nthread,
                                      DataSplitMode data_split_mode);
template SimpleDMatrix::SimpleDMatrix(CSCArrayAdapter* adapter, float missing, int nthread,
                                      DataSplitMode data_split_mode);
template SimpleDMatrix::SimpleDMatrix(CSCAdapter* adapter, float missing, int nthread,
                                      DataSplitMode data_split_mode);
template SimpleDMatrix::SimpleDMatrix(DataTableAdapter* adapter, float missing, int nthread,
                                      DataSplitMode data_split_mode);
template SimpleDMatrix::SimpleDMatrix(FileAdapter* adapter, float missing, int nthread,
                                      DataSplitMode data_split_mode);
template SimpleDMatrix::SimpleDMatrix(
    IteratorAdapter<DataIterHandle, XGBCallbackDataIterNext, XGBoostBatchCSR>* adapter,
    float missing, int nthread, DataSplitMode data_split_mode);

template <>
SimpleDMatrix::SimpleDMatrix(RecordBatchesIterAdapter* adapter, float missing, int nthread,
                             DataSplitMode data_split_mode) {
  Context ctx;
  ctx.nthread = nthread;

  auto& offset_vec = sparse_page_->offset.HostVector();
  auto& data_vec = sparse_page_->data.HostVector();
  uint64_t total_batch_size = 0;
  uint64_t total_elements = 0;

  adapter->BeforeFirst();
  // Iterate over batches of input data
  while (adapter->Next()) {
    auto& batches = adapter->Value();
    size_t num_elements = 0;
    size_t num_rows = 0;
    // Import Arrow RecordBatches
#pragma omp parallel for reduction(+ : num_elements, num_rows) num_threads(ctx.Threads())
    for (int i = 0; i < static_cast<int>(batches.size()); ++i) {  // NOLINT
      num_elements += batches[i]->Import(missing);
      num_rows += batches[i]->Size();
    }
    total_elements += num_elements;
    total_batch_size += num_rows;
    // Compute global offset for every row and starting row for every batch
    std::vector<uint64_t> batch_offsets(batches.size());
    for (size_t i = 0; i < batches.size(); ++i) {
      if (i == 0) {
        batch_offsets[i] = total_batch_size - num_rows;
        batches[i]->ShiftRowOffsets(total_elements - num_elements);
      } else {
        batch_offsets[i] = batch_offsets[i - 1] + batches[i - 1]->Size();
        batches[i]->ShiftRowOffsets(batches[i - 1]->RowOffsets().back());
      }
    }
    // Pre-allocate DMatrix memory
    data_vec.resize(total_elements);
    offset_vec.resize(total_batch_size + 1);
    // Copy data into DMatrix
#pragma omp parallel num_threads(ctx.Threads())
    {
#pragma omp for nowait
      for (int i = 0; i < static_cast<int>(batches.size()); ++i) {  // NOLINT
        size_t begin = batches[i]->RowOffsets()[0];
        for (size_t k = 0; k < batches[i]->Size(); ++k) {
          for (size_t j = 0; j < batches[i]->NumColumns(); ++j) {
            auto element = batches[i]->GetColumn(j).GetElement(k);
            if (!std::isnan(element.value)) {
              data_vec[begin++] = Entry(element.column_idx, element.value);
            }
          }
        }
      }
#pragma omp for nowait
      for (int i = 0; i < static_cast<int>(batches.size()); ++i) {
        auto& offsets = batches[i]->RowOffsets();
        std::copy(offsets.begin() + 1, offsets.end(), offset_vec.begin() + batch_offsets[i] + 1);
      }
    }
  }
  // Synchronise worker columns
  info_.num_col_ = adapter->NumColumns();
  info_.data_split_mode = data_split_mode;
  ReindexFeatures(&ctx);
  info_.SynchronizeNumberOfColumns();

  info_.num_row_ = total_batch_size;
  info_.num_nonzero_ = data_vec.size();
  CHECK_EQ(offset_vec.back(), info_.num_nonzero_);

  fmat_ctx_ = ctx;
}
}  // namespace data
}  // namespace xgboost
