/*!
 * Copyright by Contributors
 * \file predictor.h
 * \brief Interface of predictor,
 *  performs predictions for a gradient booster.
 */
#pragma once
#include <xgboost/base.h>
#include <xgboost/tree_model.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace xgboost {
class DMatrix;
}

namespace xgboost {

/**
 * \class Predictor
 *
 * \brief Performs prediction on individual training instances or batches of
 * instances.
 *
 */

class Predictor {
 public:
  virtual ~Predictor() {}

  /**
   * \fn  virtual void Predictor::PredictBatch( DMatrix* dmat, int num_feature,
   * std::vector<bst_float>* out_preds, int num_output_group, const
   * std::vector<std::unique_ptr<RegTree>>& trees, const std::vector<int>&
   * tree_info, float default_base_margin, bool init_out_predictions, int
   * tree_begin, unsigned ntree_limit = 0) = 0;
   *
   * \brief Generate batch predictions for a given feature matrix.
   *
   *
   * \param [in,out]  dmat                  Feature matrix
   * \param           num_feature           Number of features.
   * \param [in,out]  out_preds             The outUt preds.
   * \param           num_output_group      Number of output groups.
   * \param           trees                 Trees to make predictions from.
   * \param           tree_info             Information about tree group.
   * \param           default_base_margin   The default base margin.
   * \param           init_out_predictions  True to init out predictions.
   * \param           tree_begin            The tree begin index.
   * \param           ntree_limit           (Optional) The ntree limit. 0 means
   * do not limit trees.
   */

  virtual void PredictBatch(
      DMatrix* dmat, int num_feature, std::vector<bst_float>* out_preds,
      int num_output_group, const std::vector<std::unique_ptr<RegTree>>& trees,
      const std::vector<int>& tree_info, float default_base_margin,
      bool init_out_predictions, int tree_begin, unsigned ntree_limit = 0) = 0;

  /*!
   *
   * \param inst the instance you want to predict
   * \param out_preds output vector to hold the predictions
   * \param ntree_limit limit the number of trees used in prediction
   * \param root_index the root index
   * \sa Predict
   */

  /**
   * \fn  virtual void Predictor::PredictInstance( const SparseBatch::Inst&
   * inst, std::vector<bst_float>* out_preds, int num_output_group, int
   * size_leaf_vector, int num_feature, const
   * std::vector<std::unique_ptr<RegTree>>& trees, const std::vector<int>&
   * tree_info, float default_base_margin, unsigned ntree_limit = 0, unsigned
   * root_index = 0) = 0;
   *
   * \brief online prediction function, predict score for one instance at a time
   * NOTE: use the batch prediction interface if possible, batch prediction is
   * usually more efficient than online prediction This function is NOT
   * threadsafe, make sure you only call from one thread.
   *
   * \param           inst                The instance to predict.
   * \param [in,out]  out_preds           The output preds.
   * \param           num_output_group    Number of output groups.
   * \param           size_leaf_vector    The size leaf vector.
   * \param           num_feature         Number of features.
   * \param           trees               Trees to make predictions from.
   * \param           tree_info           Information about tree group.
   * \param           default_base_margin The default base margin.
   * \param           ntree_limit         (Optional) The ntree limit.
   * \param           root_index          (Optional) Zero-based index of the
   * root.
   */

  virtual void PredictInstance(
      const SparseBatch::Inst& inst, std::vector<bst_float>* out_preds,
      int num_output_group, int size_leaf_vector, int num_feature,
      const std::vector<std::unique_ptr<RegTree>>& trees,
      const std::vector<int>& tree_info, float default_base_margin,
      unsigned ntree_limit = 0, unsigned root_index = 0) = 0;
  /*!
   */

  /**
   * \fn  virtual void Predictor::PredictLeaf(DMatrix* dmat,
   * std::vector<bst_float>* out_preds, const
   * std::vector<std::unique_ptr<RegTree>>& trees, int num_features, int
   * num_output_group, unsigned ntree_limit = 0) = 0;
   *
   * \brief predict the leaf index of each tree, the output will be nsample *
   * ntree vector this is only valid in gbtree predictor.
   *
   * \param [in,out]  dmat              The input feature matrix.
   * \param [in,out]  out_preds         The output preds.
   * \param           trees             Trees to make predictions from.
   * \param           num_features      Number of features.
   * \param           num_output_group  Number of output groups.
   * \param           ntree_limit       (Optional) The ntree limit.
   */

  virtual void PredictLeaf(DMatrix* dmat, std::vector<bst_float>* out_preds,
                           const std::vector<std::unique_ptr<RegTree>>& trees,
                           int num_features, int num_output_group,
                           unsigned ntree_limit = 0) = 0;

  /**
   * \fn  virtual void Predictor::PredictContribution(DMatrix* dmat,
   * std::vector<bst_float>* out_contribs, const
   * std::vector<std::unique_ptr<RegTree>>& trees, const std::vector<int>&
   * tree_info, int num_output_group, int num_feature, float
   * default_base_margin, unsigned ntree_limit = 0) = 0;
   *
   * \brief feature contributions to individual predictions; the output will be
   * a vector of length (nfeats + 1) * num_output_group * nsample, arranged in
   * that order
   *
   * \param [in,out]  dmat              The input feature matrix.
   * \param [in,out]  out_contribs        The output feature contribs.
   * \param           trees               Trees to make predictions from.
   * \param           tree_info           Information about tree group.
   * \param           num_output_group    Number of output groups.
   * \param           num_feature         Number of features.
   * \param           default_base_margin The default base margin.
   * \param           ntree_limit         (Optional) The ntree limit.
   */

  virtual void PredictContribution(
      DMatrix* dmat, std::vector<bst_float>* out_contribs,
      const std::vector<std::unique_ptr<RegTree>>& trees,
      const std::vector<int>& tree_info, int num_output_group, int num_feature,
      float default_base_margin, unsigned ntree_limit = 0) = 0;

  /**
   * \fn  static Predictor* Predictor::Create(std::string name);
   *
   * \brief Creates a new Predictor*.
   *
   * \param name  The name.
   *
   * \return  Null if it fails, else a pointer to a Predictor.
   */

  static Predictor* Create(std::string name);
};

/*!
 * \brief Registry entry for predictor.
 */
struct PredictorReg
    : public dmlc::FunctionRegEntryBase<PredictorReg,
                                        std::function<Predictor*()>> {};

#define XGBOOST_REGISTER_PREDICTOR(UniqueId, Name)      \
  static DMLC_ATTRIBUTE_UNUSED ::xgboost::PredictorReg& \
      __make_##PredictorReg##_##UniqueId##__ =          \
          ::dmlc::Registry<::xgboost::PredictorReg>::Get()->__REGISTER__(Name)
}  // namespace xgboost
