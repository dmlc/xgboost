/*!
 * Copyright 2019 by Contributors
 * \file rv.h
 * \author Jiaming Yuan
 */

#ifndef BART_RV_H_
#define BART_RV_H_

#include <random>
#include <xgboost/base.h>
#include "../../src/common/random.h"

namespace xgboost {

class RandomVariable {
 public:
  virtual bst_float sample() const = 0;
  virtual bst_float mean() const = 0;
  virtual bst_float variance() const = 0;
};

class Gamma : public RandomVariable {
 protected:
  bst_float _shape;
  bst_float _rate;

 public:
  Gamma(bst_float shape=0, bst_float rate=0) :
      _shape{shape}, _rate{rate} {
    CHECK_GT(shape, 0.0f);
    CHECK_GT(rate, 0.0f);
  }

  bst_float sample() const override {
    // c++ parameterize as shape and scale, while in Bayes it's more
    // usual to parameterize as shape and rate
    // rate = 1 / scale
    std::gamma_distribution<bst_float> distribution(_shape, 1.0f / (_rate + kRtEps));
    return distribution(common::GlobalRandom());
  }

  bst_float mean() const override {
    return _shape / _rate;
  }
  bst_float variance() const override {
    return _shape / (_rate * _rate);
  }
};

class InverseGamma final : public Gamma {
 public:
  InverseGamma(bst_float shape=1.0, bst_float rate=1.0) : Gamma(shape, rate) {}
  bst_float sample() const override {
    bst_float inv_gamma = 1.0f / Gamma::sample();
    return inv_gamma;
  }
  bst_float mean() const override {
    bst_float scale = 1 / Gamma::_rate;
    return scale / (Gamma::_shape - 1);
  }
  bst_float variance() const override {
    bst_float scale = 1 / Gamma::_rate;
    bst_float var = (scale * scale) / ((_shape - 1) * (_shape - 1) * (_shape - 2));
    return var;
  }
};

class Normal final : public RandomVariable {
  bst_float _loc;
  bst_float _scale;

 public:
  Normal(bst_float loc=0, bst_float scale=1) : _loc{loc}, _scale{scale} {}

  bst_float mean()     const override { return _loc;   }
  bst_float variance() const override { return _scale; }

  bst_float sample() const override {
    std::normal_distribution<bst_float> distribution(_loc, _scale);
    return distribution(common::GlobalRandom());
  }
};

class Uniform final : public RandomVariable {
  bst_float _lower;
  bst_float _upper;

 public:
  Uniform(bst_float lower=0, bst_float upper=1) :
      _lower{lower}, _upper{upper} {}

  bst_float sample() const override {
    std::uniform_real_distribution<bst_float> distribution(_lower, _upper);
    return distribution(common::GlobalRandom());
  }

  bst_float mean() const override {
    return 0.5 * (_lower + _upper);
  }
  bst_float variance() const override {
    bst_float var = (1.0f / 12.0f) * (_upper - _lower) * (_upper - _lower);
    return var;
  }
};

}      // namespace xgboost
#endif  // BART_RV_H_
