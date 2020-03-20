/*!
 * Copyright 2020 by Contributors
 * \file probability_distribution.h
 * \brief Implementation of a few useful probability distributions
 * \author Avinash Barnwal and Hyunsu Cho
 */

#ifndef XGBOOST_COMMON_PROBABILITY_DISTRIBUTION_H_
#define XGBOOST_COMMON_PROBABILITY_DISTRIBUTION_H_

namespace xgboost {
namespace common {

/*! \brief Enum encoding possible choices of probability distribution */
enum class ProbabilityDistributionType : int {
  kNormal = 0, kLogistic = 1, kExtreme = 2
};

/*! \brief Constant PI */
const double kPI = 3.14159265358979323846;

class ProbabilityDistribution {
 public:
  virtual double PDF(double z) = 0;
  virtual double CDF(double z) = 0;
  virtual double GradPDF(double z) = 0;
  virtual double HessPDF(double z) = 0;

  static ProbabilityDistribution* Create(ProbabilityDistributionType dist);
};

class NormalDist : public ProbabilityDistribution {
 public:
  double PDF(double z) override;
  double CDF(double z) override;
  double GradPDF(double z) override;
  double HessPDF(double z) override;
};

class LogisticDist : public ProbabilityDistribution {
 public:
  double PDF(double z) override;
  double CDF(double z) override;
  double GradPDF(double z) override;
  double HessPDF(double z) override;
};

class ExtremeDist : public ProbabilityDistribution {
 public:
  double PDF(double z) override;
  double CDF(double z) override;
  double GradPDF(double z) override;
  double HessPDF(double z) override;
};

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_PROBABILITY_DISTRIBUTION_H_
