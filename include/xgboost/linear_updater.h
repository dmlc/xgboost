/*
 * Copyright 2018 by Contributors
 */
#pragma once

#include <dmlc/registry.h>
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/generic_parameters.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/model.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>


namespace xgboost {

class Json;

namespace gbm {
class GBLinearModel;
}  // namespace gbm

/*!
 * \brief interface of linear updater
 */
class LinearUpdater : public Configurable {
 protected:
  GenericParameter const* learner_param_;

 public:
  /*! \brief virtual destructor */
  ~LinearUpdater() override = default;
  /*!
   * \brief Initialize the updater with given arguments.
   * \param args arguments to the objective function.
   */
  virtual void Configure(
      const std::vector<std::pair<std::string, std::string> >& args) = 0;

  /**
   * \brief Updates linear model given gradients.
   *
   * \param in_gpair            The gradient pair statistics of the data.
   * \param data                Input data matrix.
   * \param model               Model to be updated.
   * \param sum_instance_weight The sum instance weights, used to normalise l1/l2 penalty.
   */
  virtual void Update(HostDeviceVector<GradientPair>* in_gpair, DMatrix* data,
                      gbm::GBLinearModel* model,
                      double sum_instance_weight) = 0;

  /*!
   * \brief Create a linear updater given name
   * \param name Name of the linear updater.
   */
  static LinearUpdater* Create(const std::string& name, GenericParameter const*);
};

/*!
 * \brief Registry entry for linear updater.
 */
struct LinearUpdaterReg
    : public dmlc::FunctionRegEntryBase<LinearUpdaterReg,
                                        std::function<LinearUpdater*()> > {};

/*!
 * \brief Macro to register linear updater.
 */
#define XGBOOST_REGISTER_LINEAR_UPDATER(UniqueId, Name)                        \
  static DMLC_ATTRIBUTE_UNUSED ::xgboost::LinearUpdaterReg&                    \
      __make_##LinearUpdaterReg##_##UniqueId##__ =                             \
          ::dmlc::Registry< ::xgboost::LinearUpdaterReg>::Get()->__REGISTER__( \
              Name)

}  // namespace xgboost
