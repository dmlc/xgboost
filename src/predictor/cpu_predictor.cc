#include <xgboost/predictor.h>
#include <xgboost/tree_model.h>
#include "dmlc/logging.h"

namespace xgboost {
namespace predictor {

class CPUPredictor : public Predictor {
  void PredictBatch(DMatrix* p_fmat, int num_feature,
                    std::vector<bst_float>* out_preds, int num_output_group,
                    const std::vector<std::unique_ptr<RegTree>>& trees,
                    const std::vector<int>& tree_info,
                    float default_base_margin, bool init_out_predictions,
                    int tree_begin, unsigned ntree_limit) override {
    PredLoopInternal(p_fmat, num_feature, out_preds, trees, tree_info,
                     tree_begin, ntree_limit, init_out_predictions,
                     num_output_group, default_base_margin);
  }

 protected:
  // internal prediction loop
  // add predictions to out_preds
  inline void PredLoopInternal(
      DMatrix* p_fmat, int num_feature, std::vector<bst_float>* out_preds,
      const std::vector<std::unique_ptr<RegTree>>& trees,
      const std::vector<int>& tree_info, unsigned tree_begin,
      unsigned ntree_limit, bool init_out_preds, int num_output_group,
      float default_base_margin) {
    int num_group = num_output_group;
    ntree_limit *= num_group;
    if (ntree_limit == 0 || ntree_limit > trees.size()) {
      ntree_limit = static_cast<unsigned>(trees.size());
    }

    if (init_out_preds) {
      size_t n = num_group * p_fmat->info().num_row;
      const std::vector<bst_float>& base_margin = p_fmat->info().base_margin;
      out_preds->resize(n);
      if (base_margin.size() != 0) {
        CHECK_EQ(out_preds->size(), n);
        std::copy(base_margin.begin(), base_margin.end(), out_preds->begin());
      } else {
        std::fill(out_preds->begin(), out_preds->end(), default_base_margin);
      }
    }

    if (num_group == 1) {
      PredLoopSpecalize(p_fmat, num_feature, out_preds, trees, tree_info, 1,
                        tree_begin, ntree_limit);
    } else {
      PredLoopSpecalize(p_fmat, num_feature, out_preds, trees, tree_info,
                        num_group, tree_begin, ntree_limit);
    }
  }

  inline void PredLoopSpecalize(
      DMatrix* p_fmat, int num_feature, std::vector<bst_float>* out_preds,
      const std::vector<std::unique_ptr<RegTree>>& trees,
      const std::vector<int>& tree_info, int num_group, unsigned tree_begin,
      unsigned tree_end) {
    const MetaInfo& info = p_fmat->info();
    const int nthread = omp_get_max_threads();
    InitThreadTemp(nthread, num_feature);
    std::vector<bst_float>& preds = *out_preds;
    // CHECK_EQ(mparam.size_leaf_vector, 0)
    //    << "size_leaf_vector is enforced to 0 so far";
    CHECK_EQ(preds.size(), p_fmat->info().num_row * num_group);
    // start collecting the prediction
    dmlc::DataIter<RowBatch>* iter = p_fmat->RowIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch& batch = iter->Value();
      // parallel over local batch
      const int K = 8;
      const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
      const bst_omp_uint rest = nsize % K;
#pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize - rest; i += K) {
        const int tid = omp_get_thread_num();
        RegTree::FVec& feats = thread_temp[tid];
        int64_t ridx[K];
        RowBatch::Inst inst[K];
        for (int k = 0; k < K; ++k) {
          ridx[k] = static_cast<int64_t>(batch.base_rowid + i + k);
        }
        for (int k = 0; k < K; ++k) {
          inst[k] = batch[i + k];
        }
        for (int k = 0; k < K; ++k) {
          for (int gid = 0; gid < num_group; ++gid) {
            const size_t offset = ridx[k] * num_group + gid;
            preds[offset] += this->PredValue(inst[k], trees, tree_info, gid,
                                             info.GetRoot(ridx[k]), &feats,
                                             tree_begin, tree_end);
          }
        }
      }
      for (bst_omp_uint i = nsize - rest; i < nsize; ++i) {
        RegTree::FVec& feats = thread_temp[0];
        const int64_t ridx = static_cast<int64_t>(batch.base_rowid + i);
        const RowBatch::Inst inst = batch[i];
        for (int gid = 0; gid < num_group; ++gid) {
          const size_t offset = ridx * num_group + gid;
          preds[offset] +=
              this->PredValue(inst, trees, tree_info, gid, info.GetRoot(ridx),
                              &feats, tree_begin, tree_end);
        }
      }
    }
  }

  void PredictInstance(const SparseBatch::Inst& inst,
                       std::vector<bst_float>* out_preds, int num_output_group,
                       int size_leaf_vector, int num_feature,
                       const std::vector<std::unique_ptr<RegTree>>& trees,
                       const std::vector<int>& tree_info,
                       float default_base_margin, unsigned ntree_limit,
                       unsigned root_index) override {
    if (thread_temp.size() == 0) {
      thread_temp.resize(1, RegTree::FVec());
      thread_temp[0].Init(num_feature);
    }
    ntree_limit *= num_output_group;
    if (ntree_limit == 0 || ntree_limit > trees.size()) {
      ntree_limit = static_cast<unsigned>(trees.size());
    }
    out_preds->resize(num_output_group * (size_leaf_vector + 1));
    // loop over output groups
    for (int gid = 0; gid < num_output_group; ++gid) {
      (*out_preds)[gid] = PredValue(inst, trees, tree_info, gid, root_index,
                                    &thread_temp[0], 0, ntree_limit) +
                          default_base_margin;
    }
  }
  void PredictLeaf(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                   std::vector<std::unique_ptr<RegTree>>& trees,
                   int num_features, int num_output_group,
                   unsigned ntree_limit) override {
    const int nthread = omp_get_max_threads();
    InitThreadTemp(nthread, num_features);
    const MetaInfo& info = p_fmat->info();
    // number of valid trees
    ntree_limit *= num_output_group;
    if (ntree_limit == 0 || ntree_limit > trees.size()) {
      ntree_limit = static_cast<unsigned>(trees.size());
    }
    std::vector<bst_float>& preds = *out_preds;
    preds.resize(info.num_row * ntree_limit);
    // start collecting the prediction
    dmlc::DataIter<RowBatch>* iter = p_fmat->RowIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch& batch = iter->Value();
      // parallel over local batch
      const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
#pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
        const int tid = omp_get_thread_num();
        size_t ridx = static_cast<size_t>(batch.base_rowid + i);
        RegTree::FVec& feats = thread_temp[tid];
        feats.Fill(batch[i]);
        for (unsigned j = 0; j < ntree_limit; ++j) {
          int tid = trees[j]->GetLeafIndex(feats, info.GetRoot(ridx));
          preds[ridx * ntree_limit + j] = static_cast<bst_float>(tid);
        }
        feats.Drop(batch[i]);
      }
    }
  }

  void PredictContribution(DMatrix* p_fmat,
                           std::vector<bst_float>* out_contribs,
                           std::vector<std::unique_ptr<RegTree>>& trees,
                           std::vector<int>& tree_info, int num_output_group,
                           int num_feature, float default_base_margin,
                           unsigned ntree_limit) override {
    const int nthread = omp_get_max_threads();
    InitThreadTemp(nthread, num_feature);
    const MetaInfo& info = p_fmat->info();
    // number of valid trees
    ntree_limit *= num_output_group;
    if (ntree_limit == 0 || ntree_limit > trees.size()) {
      ntree_limit = static_cast<unsigned>(trees.size());
    }
    const int ngroup = num_output_group;
    size_t ncolumns = num_feature + 1;
    // allocate space for (number of features + bias) times the number of rows
    std::vector<bst_float>& contribs = *out_contribs;
    contribs.resize(info.num_row * ncolumns * num_output_group);
    // make sure contributions is zeroed, we could be reusing a previously
    // allocated one
    std::fill(contribs.begin(), contribs.end(), 0);
// initialize tree node mean values
#pragma omp parallel for schedule(static)
    for (bst_omp_uint i = 0; i < ntree_limit; ++i) {
      trees[i]->FillNodeMeanValues();
    }
    // start collecting the contributions
    dmlc::DataIter<RowBatch>* iter = p_fmat->RowIterator();
    const std::vector<bst_float>& base_margin = info.base_margin;
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch& batch = iter->Value();
      // parallel over local batch
      const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
#pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
        size_t row_idx = static_cast<size_t>(batch.base_rowid + i);
        unsigned root_id = info.GetRoot(row_idx);
        RegTree::FVec& feats = thread_temp[omp_get_thread_num()];
        // loop over all classes
        for (int gid = 0; gid < ngroup; ++gid) {
          bst_float* p_contribs =
              &contribs[(row_idx * ngroup + gid) * ncolumns];
          feats.Fill(batch[i]);
          // calculate contributions
          for (unsigned j = 0; j < ntree_limit; ++j) {
            if (tree_info[j] != gid) {
              continue;
            }
            trees[j]->CalculateContributions(feats, root_id, p_contribs);
          }
          feats.Drop(batch[i]);
          // add base margin to BIAS
          if (base_margin.size() != 0) {
            p_contribs[ncolumns - 1] += base_margin[row_idx * ngroup + gid];
          } else {
            p_contribs[ncolumns - 1] += default_base_margin;
          }
        }
      }
    }
  }
  static bst_float PredValue(const RowBatch::Inst& inst,
                             const std::vector<std::unique_ptr<RegTree>>& trees,
                             const std::vector<int>& tree_info, int bst_group,
                             unsigned root_index, RegTree::FVec* p_feats,
                             unsigned tree_begin, unsigned tree_end) {
    bst_float psum = 0.0f;
    p_feats->Fill(inst);
    for (size_t i = tree_begin; i < tree_end; ++i) {
      if (tree_info[i] == bst_group) {
        int tid = trees[i]->GetLeafIndex(*p_feats, root_index);
        psum += (*trees[i])[tid].leaf_value();
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
  std::vector<RegTree::FVec> thread_temp;
};

XGBOOST_REGISTER_PREDICTOR(CPUPredictor, "cpu_predictor")
    .describe("Make predictions using CPU.")
    .set_body([]() { return new CPUPredictor(); });
}  // namespace predictor
}  // namespace xgboost
