/**
 * Copyright 2017-2025, XGBoost Contributors
 * \file predictor.h
 * \brief Interface of predictor,
 *  performs predictions for a gradient booster.
 */
#pragma once
#include <dmlc/registry.h>    // for FunctionRegEntryBase
#include <xgboost/base.h>     // for bst_tree_t
#include <xgboost/context.h>  // for Context
#include <xgboost/data.h>
#include <xgboost/host_device_vector.h>

#include <functional>  // for function
#include <string>
#include <vector>

// Forward declarations
namespace xgboost::gbm {
struct GBTreeModel;
}  // namespace xgboost::gbm

namespace xgboost {
/**
 * \class Predictor
 *
 * \brief Performs prediction on individual training instances or batches of instances for
 *        GBTree. Prediction functions all take a GBTreeModel and a DMatrix as input and
 *        output a vector of predictions. The predictor does not modify any state of the
 *        model itself.
 */
class Predictor {
 protected:
  Context const* ctx_;

 public:
  explicit Predictor(Context const* ctx) : ctx_{ctx} {}

  virtual ~Predictor() = default;

  /**
   * \brief Generate batch predictions for a given feature matrix. May use
   * cached predictions if available instead of calculating from scratch.
   *
   * \param [in,out]  dmat        Feature matrix.
   * \param [in,out]  out_preds   The output preds.
   * \param           model       The model to predict from.
   * \param           tree_begin  The tree begin index.
   * \param           tree_end    The tree end index.
   * \param           tree_weights_override Optional weights for temporary prediction overrides.
   */
  virtual void PredictBatch(DMatrix* dmat, HostDeviceVector<float>* out_preds,
                            gbm::GBTreeModel const& model, bst_tree_t tree_begin,
                            bst_tree_t tree_end = 0,
                            std::vector<float> const* tree_weights_override = nullptr) const = 0;

  /**
   * \brief Creates a new Predictor*.
   *
   * \param name  Name of the predictor.
   * \param ctx   Pointer to runtime parameters.
   */
  static Predictor* Create(std::string const& name, Context const* ctx);
};

/*!
 * \brief Registry entry for predictor.
 */
struct PredictorReg
    : public dmlc::FunctionRegEntryBase<PredictorReg, std::function<Predictor*(Context const*)>> {};

#define XGBOOST_REGISTER_PREDICTOR(UniqueId, Name)                                               \
  static DMLC_ATTRIBUTE_UNUSED ::xgboost::PredictorReg& __make_##PredictorReg##_##UniqueId##__ = \
      ::dmlc::Registry<::xgboost::PredictorReg>::Get()->__REGISTER__(Name)
}  // namespace xgboost
