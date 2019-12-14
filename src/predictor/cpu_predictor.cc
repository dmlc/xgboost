/*!
 * Copyright by Contributors 2017-2019
 */
#include <dmlc/omp.h>

#include "xgboost/predictor.h"
#include "xgboost/tree_model.h"
#include "xgboost/tree_updater.h"
#include "xgboost/logging.h"
#include "xgboost/host_device_vector.h"

#include "../gbm/gbtree_model.h"

namespace xgboost {
namespace predictor {

DMLC_REGISTRY_FILE_TAG(cpu_predictor);

class CPUPredictor : public Predictor {
 protected:
  static bst_float PredValue(const  SparsePage::Inst& inst,
                             const std::vector<std::unique_ptr<RegTree>>& trees,
                             const std::vector<int>& tree_info, int bst_group,
                             RegTree::FVec* p_feats,
                             unsigned tree_begin, unsigned tree_end) {
    bst_float psum = 0.0f;
    p_feats->Fill(inst);
    for (size_t i = tree_begin; i < tree_end; ++i) {
      if (tree_info[i] == bst_group) {
        int tid = trees[i]->GetLeafIndex(*p_feats);
        psum += (*trees[i])[tid].LeafValue();
      }
    }
    p_feats->Drop(inst);
    return psum;
  }

  // init thread buffers
  inline void InitThreadTemp(int nthread, int num_feature) {
    int prev_thread_temp_size = thread_temp.size();
    if (prev_thread_temp_size < nthread) {
      thread_temp.resize(nthread, RegTree::FVec());
      for (int i = prev_thread_temp_size; i < nthread; ++i) {
        thread_temp[i].Init(num_feature);
      }
    }
  }

  void PredLoopInternal(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                        gbm::GBTreeModel const& model, int32_t tree_begin,
                        int32_t tree_end) {
    int32_t const num_group = model.learner_model_param_->num_output_group;
    const int nthread = omp_get_max_threads();
    InitThreadTemp(nthread, model.learner_model_param_->num_feature);
    std::vector<bst_float>& preds = *out_preds;
    CHECK_EQ(model.param.size_leaf_vector, 0)
        << "size_leaf_vector is enforced to 0 so far";
    CHECK_EQ(preds.size(), p_fmat->Info().num_row_ * num_group);
    // start collecting the prediction
    for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
      // parallel over local batch
      constexpr int kUnroll = 8;
      const auto nsize = static_cast<bst_omp_uint>(batch.Size());
      const bst_omp_uint rest = nsize % kUnroll;
      // Pull to host before entering omp block, as this is not thread safe.
      batch.data.HostVector();
      batch.offset.HostVector();
      if (nsize >= kUnroll) {
#pragma omp parallel for schedule(static)
        for (bst_omp_uint i = 0; i < nsize - rest; i += kUnroll) {
          const int tid = omp_get_thread_num();
          RegTree::FVec& feats = thread_temp[tid];
          int64_t ridx[kUnroll];
          SparsePage::Inst inst[kUnroll];
          for (int k = 0; k < kUnroll; ++k) {
            ridx[k] = static_cast<int64_t>(batch.base_rowid + i + k);
          }
          for (int k = 0; k < kUnroll; ++k) {
            inst[k] = batch[i + k];
          }
          for (int k = 0; k < kUnroll; ++k) {
            for (int gid = 0; gid < num_group; ++gid) {
              const size_t offset = ridx[k] * num_group + gid;
              preds[offset] += this->PredValue(
                  inst[k], model.trees, model.tree_info, gid,
                  &feats, tree_begin, tree_end);
            }
          }
        }
      }
      for (bst_omp_uint i = nsize - rest; i < nsize; ++i) {
        RegTree::FVec& feats = thread_temp[0];
        const auto ridx = static_cast<int64_t>(batch.base_rowid + i);
        auto inst = batch[i];
        for (int gid = 0; gid < num_group; ++gid) {
          const size_t offset = ridx * num_group + gid;
          preds[offset] +=
              this->PredValue(inst, model.trees, model.tree_info, gid,
                              &feats, tree_begin, tree_end);
        }
      }
    }
  }

  bool PredictFromCache(DMatrix* dmat,
                        HostDeviceVector<bst_float>* out_preds,
                        const gbm::GBTreeModel& model,
                        unsigned ntree_limit) const {
    CHECK(cache_);
    if (ntree_limit == 0 ||
        ntree_limit * model.learner_model_param_->num_output_group >= model.trees.size()) {
      auto it = cache_->find(dmat);
      if (it != cache_->end()) {
        const HostDeviceVector<bst_float>& y = it->second.predictions;
        if (y.Size() != 0) {
          out_preds->Resize(y.Size());
          std::copy(y.HostVector().begin(), y.HostVector().end(),
                    out_preds->HostVector().begin());
          return true;
        }
      }
    }
    return false;
  }

  void InitOutPredictions(const MetaInfo& info,
                          HostDeviceVector<bst_float>* out_preds,
                          const gbm::GBTreeModel& model) const {
    CHECK_NE(model.learner_model_param_->num_output_group, 0);
    size_t n = model.learner_model_param_->num_output_group * info.num_row_;
    const auto& base_margin = info.base_margin_.HostVector();
    out_preds->Resize(n);
    std::vector<bst_float>& out_preds_h = out_preds->HostVector();
    if (base_margin.size() == n) {
      CHECK_EQ(out_preds->Size(), n);
      std::copy(base_margin.begin(), base_margin.end(), out_preds_h.begin());
    } else {
      if (!base_margin.empty()) {
        std::ostringstream oss;
        oss << "Ignoring the base margin, since it has incorrect length. "
            << "The base margin must be an array of length ";
        if (model.learner_model_param_->num_output_group > 1) {
          oss << "[num_class] * [number of data points], i.e. "
              << model.learner_model_param_->num_output_group << " * " << info.num_row_
              << " = " << n << ". ";
        } else {
          oss << "[number of data points], i.e. " << info.num_row_ << ". ";
        }
        oss << "Instead, all data points will use "
            << "base_score = " << model.learner_model_param_->base_score;
        LOG(WARNING) << oss.str();
      }
      std::fill(out_preds_h.begin(), out_preds_h.end(),
                model.learner_model_param_->base_score);
    }
  }

 public:
  CPUPredictor(GenericParameter const* generic_param,
               std::shared_ptr<std::unordered_map<DMatrix*, PredictionCacheEntry>> cache) :
      Predictor::Predictor{generic_param, cache} {}
  void PredictBatch(DMatrix* dmat, HostDeviceVector<bst_float>* out_preds,
                    const gbm::GBTreeModel& model, int tree_begin,
                    unsigned ntree_limit = 0) override {
    if (this->PredictFromCache(dmat, out_preds, model, ntree_limit)) {
      return;
    }
    this->InitOutPredictions(dmat->Info(), out_preds, model);

    ntree_limit *= model.learner_model_param_->num_output_group;
    if (ntree_limit == 0 || ntree_limit > model.trees.size()) {
      ntree_limit = static_cast<unsigned>(model.trees.size());
    }

    this->PredLoopInternal(dmat, &out_preds->HostVector(), model,
                           tree_begin, ntree_limit);

    auto cache_emtry = this->FindCache(dmat);
    if (cache_emtry == cache_->cend()) { return; }
    if (cache_emtry->second.predictions.Size() == 0) {
      // See comment in GPUPredictor::PredictBatch.
      InitOutPredictions(cache_emtry->second.data->Info(),
                         &(cache_emtry->second.predictions), model);
      cache_emtry->second.predictions.Copy(*out_preds);
    }
  }

  void UpdatePredictionCache(
      const gbm::GBTreeModel& model,
      std::vector<std::unique_ptr<TreeUpdater>>* updaters,
      int num_new_trees) override {
    int old_ntree = model.trees.size() - num_new_trees;
    // update cache entry
    for (auto& kv : (*cache_)) {
      PredictionCacheEntry& e = kv.second;

      if (e.predictions.Size() == 0) {
        InitOutPredictions(e.data->Info(), &(e.predictions), model);
        PredLoopInternal(e.data.get(), &(e.predictions.HostVector()), model, 0,
                         model.trees.size());
      } else if (model.learner_model_param_->num_output_group == 1 && updaters->size() > 0 &&
                 num_new_trees == 1 &&
                 updaters->back()->UpdatePredictionCache(e.data.get(),
                                                         &(e.predictions))) {
        {}  // do nothing
      } else {
        PredLoopInternal(e.data.get(), &(e.predictions.HostVector()), model, old_ntree,
                         model.trees.size());
      }
    }
  }

  void PredictInstance(const SparsePage::Inst& inst,
                       std::vector<bst_float>* out_preds,
                       const gbm::GBTreeModel& model, unsigned ntree_limit) override {
    if (thread_temp.size() == 0) {
      thread_temp.resize(1, RegTree::FVec());
      thread_temp[0].Init(model.learner_model_param_->num_feature);
    }
    ntree_limit *= model.learner_model_param_->num_output_group;
    if (ntree_limit == 0 || ntree_limit > model.trees.size()) {
      ntree_limit = static_cast<unsigned>(model.trees.size());
    }
    out_preds->resize(model.learner_model_param_->num_output_group *
                      (model.param.size_leaf_vector + 1));
    // loop over output groups
    for (uint32_t gid = 0; gid < model.learner_model_param_->num_output_group; ++gid) {
      (*out_preds)[gid] =
          PredValue(inst, model.trees, model.tree_info, gid,
                    &thread_temp[0], 0, ntree_limit) +
          model.learner_model_param_->base_score;
    }
  }
  void PredictLeaf(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                   const gbm::GBTreeModel& model, unsigned ntree_limit) override {
    const int nthread = omp_get_max_threads();
    InitThreadTemp(nthread, model.learner_model_param_->num_feature);
    const MetaInfo& info = p_fmat->Info();
    // number of valid trees
    ntree_limit *= model.learner_model_param_->num_output_group;
    if (ntree_limit == 0 || ntree_limit > model.trees.size()) {
      ntree_limit = static_cast<unsigned>(model.trees.size());
    }
    std::vector<bst_float>& preds = *out_preds;
    preds.resize(info.num_row_ * ntree_limit);
    // start collecting the prediction
    for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
      // parallel over local batch
      const auto nsize = static_cast<bst_omp_uint>(batch.Size());
#pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
        const int tid = omp_get_thread_num();
        auto ridx = static_cast<size_t>(batch.base_rowid + i);
        RegTree::FVec& feats = thread_temp[tid];
        feats.Fill(batch[i]);
        for (unsigned j = 0; j < ntree_limit; ++j) {
          int tid = model.trees[j]->GetLeafIndex(feats);
          preds[ridx * ntree_limit + j] = static_cast<bst_float>(tid);
        }
        feats.Drop(batch[i]);
      }
    }
  }

  void PredictContribution(DMatrix* p_fmat, std::vector<bst_float>* out_contribs,
                           const gbm::GBTreeModel& model, uint32_t ntree_limit,
                           std::vector<bst_float>* tree_weights,
                           bool approximate, int condition,
                           unsigned condition_feature) override {
    const int nthread = omp_get_max_threads();
    InitThreadTemp(nthread,  model.learner_model_param_->num_feature);
    const MetaInfo& info = p_fmat->Info();
    // number of valid trees
    ntree_limit *= model.learner_model_param_->num_output_group;
    if (ntree_limit == 0 || ntree_limit > model.trees.size()) {
      ntree_limit = static_cast<unsigned>(model.trees.size());
    }
    const int ngroup = model.learner_model_param_->num_output_group;
    CHECK_NE(ngroup, 0);
    size_t const ncolumns = model.learner_model_param_->num_feature + 1;
    CHECK_NE(ncolumns, 0);
    // allocate space for (number of features + bias) times the number of rows
    std::vector<bst_float>& contribs = *out_contribs;
    contribs.resize(info.num_row_ * ncolumns * model.learner_model_param_->num_output_group);
    // make sure contributions is zeroed, we could be reusing a previously
    // allocated one
    std::fill(contribs.begin(), contribs.end(), 0);
    // initialize tree node mean values
    #pragma omp parallel for schedule(static)
    for (bst_omp_uint i = 0; i < ntree_limit; ++i) {
      model.trees[i]->FillNodeMeanValues();
    }
    const std::vector<bst_float>& base_margin = info.base_margin_.HostVector();
    // start collecting the contributions
    for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
      // parallel over local batch
      const auto nsize = static_cast<bst_omp_uint>(batch.Size());
#pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
        auto row_idx = static_cast<size_t>(batch.base_rowid + i);
        RegTree::FVec& feats = thread_temp[omp_get_thread_num()];
        std::vector<bst_float> this_tree_contribs(ncolumns);
        // loop over all classes
        for (int gid = 0; gid < ngroup; ++gid) {
          bst_float* p_contribs = &contribs[(row_idx * ngroup + gid) * ncolumns];
          feats.Fill(batch[i]);
          // calculate contributions
          for (unsigned j = 0; j < ntree_limit; ++j) {
            std::fill(this_tree_contribs.begin(), this_tree_contribs.end(), 0);
            if (model.tree_info[j] != gid) {
              continue;
            }
            if (!approximate) {
              model.trees[j]->CalculateContributions(feats, &this_tree_contribs[0],
                                                     condition, condition_feature);
            } else {
              model.trees[j]->CalculateContributionsApprox(feats, &this_tree_contribs[0]);
            }
            for (size_t ci = 0 ; ci < ncolumns ; ++ci) {
                p_contribs[ci] += this_tree_contribs[ci] *
                    (tree_weights == nullptr ? 1 : (*tree_weights)[j]);
            }
          }
          feats.Drop(batch[i]);
          // add base margin to BIAS
          if (base_margin.size() != 0) {
            p_contribs[ncolumns - 1] += base_margin[row_idx * ngroup + gid];
          } else {
            p_contribs[ncolumns - 1] += model.learner_model_param_->base_score;
          }
        }
      }
    }
  }

  void PredictInteractionContributions(DMatrix* p_fmat, std::vector<bst_float>* out_contribs,
                                       const gbm::GBTreeModel& model, unsigned ntree_limit,
                                       std::vector<bst_float>* tree_weights,
                                       bool approximate) override {
    const MetaInfo& info = p_fmat->Info();
    const int ngroup = model.learner_model_param_->num_output_group;
    size_t const ncolumns = model.learner_model_param_->num_feature;
    const unsigned row_chunk = ngroup * (ncolumns + 1) * (ncolumns + 1);
    const unsigned mrow_chunk = (ncolumns + 1) * (ncolumns + 1);
    const unsigned crow_chunk = ngroup * (ncolumns + 1);

    // allocate space for (number of features^2) times the number of rows and tmp off/on contribs
    std::vector<bst_float>& contribs = *out_contribs;
    contribs.resize(info.num_row_ * ngroup * (ncolumns + 1) * (ncolumns + 1));
    std::vector<bst_float> contribs_off(info.num_row_ * ngroup * (ncolumns + 1));
    std::vector<bst_float> contribs_on(info.num_row_ * ngroup * (ncolumns + 1));
    std::vector<bst_float> contribs_diag(info.num_row_ * ngroup * (ncolumns + 1));

    // Compute the difference in effects when conditioning on each of the features on and off
    // see: Axiomatic characterizations of probabilistic and
    //      cardinal-probabilistic interaction indices
    PredictContribution(p_fmat, &contribs_diag, model, ntree_limit,
                        tree_weights, approximate, 0, 0);
    for (size_t i = 0; i < ncolumns + 1; ++i) {
      PredictContribution(p_fmat, &contribs_off, model, ntree_limit,
                          tree_weights, approximate, -1, i);
      PredictContribution(p_fmat, &contribs_on, model, ntree_limit,
                          tree_weights, approximate, 1, i);

      for (size_t j = 0; j < info.num_row_; ++j) {
        for (int l = 0; l < ngroup; ++l) {
          const unsigned o_offset = j * row_chunk + l * mrow_chunk + i * (ncolumns + 1);
          const unsigned c_offset = j * crow_chunk + l * (ncolumns + 1);
          contribs[o_offset + i] = 0;
          for (size_t k = 0; k < ncolumns + 1; ++k) {
            // fill in the diagonal with additive effects, and off-diagonal with the interactions
            if (k == i) {
              contribs[o_offset + i] += contribs_diag[c_offset + k];
            } else {
              contribs[o_offset + k] = (contribs_on[c_offset + k] - contribs_off[c_offset + k])/2.0;
              contribs[o_offset + i] -= contribs[o_offset + k];
            }
          }
        }
      }
    }
  }
  std::vector<RegTree::FVec> thread_temp;
};

XGBOOST_REGISTER_PREDICTOR(CPUPredictor, "cpu_predictor")
.describe("Make predictions using CPU.")
.set_body([](GenericParameter const* generic_param,
             std::shared_ptr<std::unordered_map<DMatrix*, PredictionCacheEntry>> cache) {
            return new CPUPredictor(generic_param, cache);
          });
}  // namespace predictor
}  // namespace xgboost
