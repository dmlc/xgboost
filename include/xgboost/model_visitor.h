/*!
 * Copyright 2018 by Contributors
 * \file model_visitor.h
 * \brief a generic model visitor interface
 */
#ifndef XGBOOST_MODEL_VISITOR_H_
#define XGBOOST_MODEL_VISITOR_H_

namespace xgboost {

template<typename T>
class Visitor {
    public:
        virtual void Visit(T& t) {
            // noop
        }
};

// Forward definitions
class Learner;
class GradientBooster;
namespace gbm {
	class GBTreeModel;
    class GBLinearModel;
}

class ModelVisitor : public Visitor<Learner>,
                   public Visitor<GradientBooster>,
                   public Visitor<gbm::GBTreeModel>,
                   public Visitor<gbm::GBLinearModel> {
  public:
	using Visitor<Learner>::Visit;
	using Visitor<GradientBooster>::Visit;
	using Visitor<gbm::GBTreeModel>::Visit;
	using Visitor<gbm::GBLinearModel>::Visit;
};

} // namespace xgboost

#endif
