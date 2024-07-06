/*!
 * Copyright 2017-2024 by Contributors
 * \file hist_updater.cc
 */

#include "hist_updater.h"

#include <oneapi/dpl/random>

#include "../common/hist_util.h"
#include "../../src/collective/allreduce.h"

namespace xgboost {
namespace sycl {
namespace tree {

template <typename GradientSumT>
void HistUpdater<GradientSumT>::SetHistSynchronizer(
    HistSynchronizer<GradientSumT> *sync) {
  hist_synchronizer_.reset(sync);
}

template <typename GradientSumT>
void HistUpdater<GradientSumT>::SetHistRowsAdder(
    HistRowsAdder<GradientSumT> *adder) {
  hist_rows_adder_.reset(adder);
}

template <typename GradientSumT>
void HistUpdater<GradientSumT>::BuildHistogramsLossGuide(
    ExpandEntry entry,
    const common::GHistIndexMatrix &gmat,
    RegTree *p_tree,
    const USMVector<GradientPair, MemoryType::on_device> &gpair_device) {
  nodes_for_explicit_hist_build_.clear();
  nodes_for_subtraction_trick_.clear();
  nodes_for_explicit_hist_build_.push_back(entry);

  if (!(*p_tree)[entry.nid].IsRoot()) {
    auto sibling_id = entry.GetSiblingId(p_tree);
    nodes_for_subtraction_trick_.emplace_back(sibling_id, p_tree->GetDepth(sibling_id));
  }

  std::vector<int> sync_ids;
  hist_rows_adder_->AddHistRows(this, &sync_ids, p_tree);
  qu_.wait_and_throw();
  BuildLocalHistograms(gmat, p_tree, gpair_device);
  hist_synchronizer_->SyncHistograms(this, sync_ids, p_tree);
}

template<typename GradientSumT>
void HistUpdater<GradientSumT>::BuildLocalHistograms(
    const common::GHistIndexMatrix &gmat,
    RegTree *p_tree,
    const USMVector<GradientPair, MemoryType::on_device> &gpair_device) {
  builder_monitor_.Start("BuildLocalHistograms");
  const size_t n_nodes = nodes_for_explicit_hist_build_.size();
  ::sycl::event event;

  for (size_t i = 0; i < n_nodes; i++) {
    const int32_t nid = nodes_for_explicit_hist_build_[i].nid;

    if (row_set_collection_[nid].Size() > 0) {
      event = BuildHist(gpair_device, row_set_collection_[nid], gmat, &(hist_[nid]),
                        &(hist_buffer_.GetDeviceBuffer()), event);
    } else {
      common::InitHist(qu_, &(hist_[nid]), hist_[nid].Size(), &event);
    }
  }
  qu_.wait_and_throw();
  builder_monitor_.Stop("BuildLocalHistograms");
}

template<typename GradientSumT>
void HistUpdater<GradientSumT>::InitSampling(
      const USMVector<GradientPair, MemoryType::on_device> &gpair,
      USMVector<size_t, MemoryType::on_device>* row_indices) {
  const size_t num_rows = row_indices->Size();
  auto* row_idx = row_indices->Data();
  const auto* gpair_ptr = gpair.DataConst();
  uint64_t num_samples = 0;
  const auto subsample = param_.subsample;
  ::sycl::event event;

  {
    ::sycl::buffer<uint64_t, 1> flag_buf(&num_samples, 1);
    uint64_t seed = seed_;
    seed_ += num_rows;
    event = qu_.submit([&](::sycl::handler& cgh) {
      auto flag_buf_acc  = flag_buf.get_access<::sycl::access::mode::read_write>(cgh);
      cgh.parallel_for<>(::sycl::range<1>(::sycl::range<1>(num_rows)),
                                          [=](::sycl::item<1> pid) {
        uint64_t i = pid.get_id(0);

        // Create minstd_rand engine
        oneapi::dpl::minstd_rand engine(seed, i);
        oneapi::dpl::bernoulli_distribution coin_flip(subsample);

        auto rnd = coin_flip(engine);
        if (gpair_ptr[i].GetHess() >= 0.0f && rnd) {
          AtomicRef<uint64_t> num_samples_ref(flag_buf_acc[0]);
          row_idx[num_samples_ref++] = i;
        }
      });
    });
    /* After calling a destructor for flag_buf,  content will be copyed to num_samples */
  }

  row_indices->Resize(&qu_, num_samples, 0, &event);
  qu_.wait();
}

template<typename GradientSumT>
void HistUpdater<GradientSumT>::InitData(
                                const common::GHistIndexMatrix& gmat,
                                const USMVector<GradientPair, MemoryType::on_device> &gpair,
                                const DMatrix& fmat,
                                const RegTree& tree) {
  CHECK((param_.max_depth > 0 || param_.max_leaves > 0))
      << "max_depth or max_leaves cannot be both 0 (unlimited); "
      << "at least one should be a positive quantity.";
  if (param_.grow_policy == xgboost::tree::TrainParam::kDepthWise) {
    CHECK(param_.max_depth > 0) << "max_depth cannot be 0 (unlimited) "
                                << "when grow_policy is depthwise.";
  }
  builder_monitor_.Start("InitData");
  const auto& info = fmat.Info();

  // initialize the row set
  {
    row_set_collection_.Clear();

    // initialize histogram collection
    uint32_t nbins = gmat.cut.Ptrs().back();
    hist_.Init(qu_, nbins);

    hist_buffer_.Init(qu_, nbins);
    size_t buffer_size = kBufferSize;
    hist_buffer_.Reset(kBufferSize);

    // initialize histogram builder
    hist_builder_ = common::GHistBuilder<GradientSumT>(qu_, nbins);

    USMVector<size_t, MemoryType::on_device>* row_indices = &(row_set_collection_.Data());
    row_indices->Resize(&qu_, info.num_row_);
    size_t* p_row_indices = row_indices->Data();
    // mark subsample and build list of member rows
    if (param_.subsample < 1.0f) {
      CHECK_EQ(param_.sampling_method, xgboost::tree::TrainParam::kUniform)
        << "Only uniform sampling is supported, "
        << "gradient-based sampling is only support by GPU Hist.";
      InitSampling(gpair, row_indices);
    } else {
      int has_neg_hess = 0;
      const GradientPair* gpair_ptr = gpair.DataConst();
      ::sycl::event event;
      {
        ::sycl::buffer<int, 1> flag_buf(&has_neg_hess, 1);
        event = qu_.submit([&](::sycl::handler& cgh) {
          auto flag_buf_acc  = flag_buf.get_access<::sycl::access::mode::read_write>(cgh);
          cgh.parallel_for<>(::sycl::range<1>(::sycl::range<1>(info.num_row_)),
                                            [=](::sycl::item<1> pid) {
            const size_t idx = pid.get_id(0);
            p_row_indices[idx] = idx;
            if (gpair_ptr[idx].GetHess() < 0.0f) {
              AtomicRef<int> has_neg_hess_ref(flag_buf_acc[0]);
              has_neg_hess_ref.fetch_max(1);
            }
          });
        });
      }

      if (has_neg_hess) {
        size_t max_idx = 0;
        {
          ::sycl::buffer<size_t, 1> flag_buf(&max_idx, 1);
          event = qu_.submit([&](::sycl::handler& cgh) {
            cgh.depends_on(event);
            auto flag_buf_acc  = flag_buf.get_access<::sycl::access::mode::read_write>(cgh);
            cgh.parallel_for<>(::sycl::range<1>(::sycl::range<1>(info.num_row_)),
                                                [=](::sycl::item<1> pid) {
              const size_t idx = pid.get_id(0);
              if (gpair_ptr[idx].GetHess() >= 0.0f) {
                AtomicRef<size_t> max_idx_ref(flag_buf_acc[0]);
                p_row_indices[max_idx_ref++] = idx;
              }
            });
          });
        }
        row_indices->Resize(&qu_, max_idx, 0, &event);
      }
      qu_.wait_and_throw();
    }
  }
  row_set_collection_.Init();

  {
    /* determine layout of data */
    const size_t nrow = info.num_row_;
    const size_t ncol = info.num_col_;
    const size_t nnz = info.num_nonzero_;
    // number of discrete bins for feature 0
    const uint32_t nbins_f0 = gmat.cut.Ptrs()[1] - gmat.cut.Ptrs()[0];
    if (nrow * ncol == nnz) {
      // dense data with zero-based indexing
      data_layout_ = kDenseDataZeroBased;
    } else if (nbins_f0 == 0 && nrow * (ncol - 1) == nnz) {
      // dense data with one-based indexing
      data_layout_ = kDenseDataOneBased;
    } else {
      // sparse data
      data_layout_ = kSparseData;
    }
  }

  if (data_layout_ == kDenseDataZeroBased || data_layout_ == kDenseDataOneBased) {
    /* specialized code for dense data:
       choose the column that has a least positive number of discrete bins.
       For dense data (with no missing value),
       the sum of gradient histogram is equal to snode[nid] */
    const std::vector<uint32_t>& row_ptr = gmat.cut.Ptrs();
    const auto nfeature = static_cast<bst_uint>(row_ptr.size() - 1);
    uint32_t min_nbins_per_feature = 0;
    for (bst_uint i = 0; i < nfeature; ++i) {
      const uint32_t nbins = row_ptr[i + 1] - row_ptr[i];
      if (nbins > 0) {
        if (min_nbins_per_feature == 0 || min_nbins_per_feature > nbins) {
          min_nbins_per_feature = nbins;
          fid_least_bins_ = i;
        }
      }
    }
    CHECK_GT(min_nbins_per_feature, 0U);
  }

  std::fill(snode_host_.begin(), snode_host_.end(),  NodeEntry<GradientSumT>(param_));
  builder_monitor_.Stop("InitData");
}

template <typename GradientSumT>
void HistUpdater<GradientSumT>::InitNewNode(int nid,
                                            const common::GHistIndexMatrix& gmat,
                                            const USMVector<GradientPair,
                                                            MemoryType::on_device> &gpair,
                                            const DMatrix& fmat,
                                            const RegTree& tree) {
  builder_monitor_.Start("InitNewNode");

  snode_host_.resize(tree.NumNodes(), NodeEntry<GradientSumT>(param_));
  {
    if (tree[nid].IsRoot()) {
      GradStats<GradientSumT> grad_stat;
      if (data_layout_ == kDenseDataZeroBased || data_layout_ == kDenseDataOneBased) {
        const std::vector<uint32_t>& row_ptr = gmat.cut.Ptrs();
        const uint32_t ibegin = row_ptr[fid_least_bins_];
        const uint32_t iend = row_ptr[fid_least_bins_ + 1];
        const auto* hist = reinterpret_cast<GradStats<GradientSumT>*>(hist_[nid].Data());

        std::vector<GradStats<GradientSumT>> ets(iend - ibegin);
        qu_.memcpy(ets.data(), hist + ibegin,
                   (iend - ibegin) * sizeof(GradStats<GradientSumT>)).wait_and_throw();
        for (const auto& et : ets) {
          grad_stat += et;
        }
      } else {
        const common::RowSetCollection::Elem e = row_set_collection_[nid];
        const size_t* row_idxs = e.begin;
        const size_t size = e.Size();
        const GradientPair* gpair_ptr = gpair.DataConst();

        ::sycl::buffer<GradStats<GradientSumT>> buff(&grad_stat, 1);
        qu_.submit([&](::sycl::handler& cgh) {
          auto reduction = ::sycl::reduction(buff, cgh, ::sycl::plus<>());
          cgh.parallel_for<>(::sycl::range<1>(size), reduction,
                            [=](::sycl::item<1> pid, auto& sum) {
            size_t i = pid.get_id(0);
            size_t row_idx = row_idxs[i];
            if constexpr (std::is_same<GradientPair::ValueT, GradientSumT>::value) {
              sum += gpair_ptr[row_idx];
            } else {
              sum += GradStats<GradientSumT>(gpair_ptr[row_idx].GetGrad(),
                                             gpair_ptr[row_idx].GetHess());
            }
          });
        }).wait_and_throw();
      }
      auto rc = collective::Allreduce(
                      ctx_, linalg::MakeVec(reinterpret_cast<GradientSumT*>(&grad_stat), 2),
                      collective::Op::kSum);
      SafeColl(rc);
      snode_host_[nid].stats = grad_stat;
    } else {
      int parent_id = tree[nid].Parent();
      if (tree[nid].IsLeftChild()) {
        snode_host_[nid].stats = snode_host_[parent_id].best.left_sum;
      } else {
        snode_host_[nid].stats = snode_host_[parent_id].best.right_sum;
      }
    }
  }

  // calculating the weights
  {
    auto evaluator = tree_evaluator_.GetEvaluator();
    bst_uint parentid = tree[nid].Parent();
    snode_host_[nid].weight = evaluator.CalcWeight(parentid, snode_host_[nid].stats);
    snode_host_[nid].root_gain = evaluator.CalcGain(parentid, snode_host_[nid].stats);
  }
  builder_monitor_.Stop("InitNewNode");
}

template class HistUpdater<float>;
template class HistUpdater<double>;

}  // namespace tree
}  // namespace sycl
}  // namespace xgboost
