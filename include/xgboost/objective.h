/**
 * Copyright 2014-2026, XGBoost Contributors
 *
 * @brief interface of objective function used by xgboost.
 * @author Tianqi Chen, Kailong Chen
 */
#ifndef XGBOOST_OBJECTIVE_H_
#define XGBOOST_OBJECTIVE_H_

#include <dmlc/registry.h>
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/linalg.h>  // for Vector
#include <xgboost/model.h>
#include <xgboost/task.h>

#include <cstdint>  // for int32_t
#include <functional>
#include <set>
#include <string>   // for string
#include <utility>  // for move
#include <vector>   // for vector

namespace xgboost {

struct Context;

/** @brief The interface of objective function */
class ObjFunction : public Configurable {
 protected:
  Context const* ctx_{nullptr};

 public:
  static constexpr float DefaultBaseScore() { return 0.5f; }

 public:
  ~ObjFunction() override = default;
  /**
   * @brief Configure the objective with the specified parameters.
   *
   * @param args arguments to the objective function.
   * @return Names of parameters consumed by the objective.
   */
  virtual std::set<std::string> Configure(Args const& args) = 0;
  /**
   * @brief Get gradient over each of predictions, given existing information.
   *
   * @param preds Raw prediction (before applying the inverse link) of the current round.
   * @param info information about labels, weights, groups in rank.
   * @param iteration current iteration number.
   * @param out_gpair output of get gradient, saves gradient and second order gradient in
   */
  virtual void GetGradient(HostDeviceVector<float> const& preds, MetaInfo const& info,
                           std::int32_t iter, linalg::Matrix<GradientPair>* out_gpair) = 0;

  /** @return the default evaluation metric for the objective */
  [[nodiscard]] virtual const char* DefaultEvalMetric() const = 0;
  /**
   * @brief Return the configuration for the default metric.
   */
  [[nodiscard]] virtual Json DefaultMetricConfig() const { return Json{Null{}}; }
  /**
   * @brief Apply inverse link (activation) function to prediction values.
   *
   *   This is only called when Prediction is called
   *
   * @param [in,out] io_preds prediction values, saves to this vector as well.
   * @param shape (rows, outputs) shape before transformation.
   * @return Shape `(rows, transformed_outputs)` of the row-major buffer in io_preds.
   *         The row count is unchanged; transforms that change the output width must
   *         return the new width even for zero rows. Singleton axes are retained.
   */
  virtual std::vector<bst_ulong> PredTransform(HostDeviceVector<float>*,
                                               std::vector<bst_ulong> shape) const {
    return shape;
  }
  /**
   * @brief Apply inverse link (activation) function to prediction values
   *
   *  This is only called when Eval is called, usually it redirects to PredTransform.
   *
   * @param [in,out] io_preds Prediction values, overwritten with the metric inputs.
   * @param shape `(rows, outputs)` shape before transformation.
   * @return Shape `(rows, metric_outputs)` of the row-major buffer in io_preds after
   *         evaluation transformation, including for empty input. Singleton axes are
   *         retained. This can differ from PredTransform, e.g. softmax metrics use class
   *         probabilities instead of class indices.
   */
  virtual std::vector<bst_ulong> EvalTransform(HostDeviceVector<float>* io_preds,
                                               std::vector<bst_ulong> shape) {
    return this->PredTransform(io_preds, std::move(shape));
  }
  /**
   * @brief Apply the link function to the intercept.
   *
   *   This is an inverse of `PredTransform` for most of the objectives (if there's a
   *   valid inverse). It's used to transform user-set base_score back to margin used by
   *   gradient boosting. The method converts objective-based valid outputs like
   *   probability back to raw model outputs.
   *
   * @param [in,out] base_score The intercept to transform.
   */
  virtual void ProbToMargin(linalg::Vector<float>* /*base_score*/) const {}
  /**
   * @brief Obtain the initial estimation of prediction (intercept).
   *
   *   The output in `base_score` represents prediction after applying the inverse link
   *   function (valid prediction instead of raw).
   *
   * @param info MetaInfo that contains label.
   * @param base_score Output estimation.
   */
  virtual void InitEstimation(MetaInfo const& info, linalg::Vector<float>* base_score) const;
  /**
   * @brief Return task of this objective.
   */
  [[nodiscard]] virtual struct ObjInfo Task() const = 0;
  /**
   * @brief Return number of targets for input matrix.  Right now XGBoost supports only
   *        multi-target regression.
   */
  [[nodiscard]] virtual bst_target_t Targets(MetaInfo const& info) const {
    if (info.labels.Shape(1) > 1) {
      LOG(FATAL) << "multioutput is not supported by the current objective function";
    }
    return 1;
  }
  /** @brief Getter of the context. */
  [[nodiscard]] Context const* Ctx() const { return this->ctx_; }

  /**
   * @brief Create an objective function according to the name.
   *
   * @param name Name of the objective.
   * @param ctx  Pointer to the context.
   */
  static ObjFunction* Create(const std::string& name, Context const* ctx);
};

/*!
 * \brief Registry entry for objective factory functions.
 */
struct ObjFunctionReg
    : public dmlc::FunctionRegEntryBase<ObjFunctionReg, std::function<ObjFunction*()> > {};

/*!
 * \brief Macro to register objective function.
 *
 * \code
 * // example of registering a objective
 * XGBOOST_REGISTER_OBJECTIVE(LinearRegression, "reg:squarederror")
 * .describe("Linear regression objective")
 * .set_body([]() {
 *     return new RegLossObj(LossType::kLinearSquare);
 *   });
 * \endcode
 */
#define XGBOOST_REGISTER_OBJECTIVE(UniqueId, Name)        \
  static DMLC_ATTRIBUTE_UNUSED ::xgboost::ObjFunctionReg& \
      __make_##ObjFunctionReg##_##UniqueId##__ =          \
          ::dmlc::Registry< ::xgboost::ObjFunctionReg>::Get()->__REGISTER__(Name)
}  // namespace xgboost
#endif  // XGBOOST_OBJECTIVE_H_
