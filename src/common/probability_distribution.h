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

namespace probability_constant {

/*! \brief Constant PI */
const double kPI = 3.14159265358979323846;
/*! \brief The Euler-Mascheroni_constant */
const double kEulerMascheroni = 0.57721566490153286060651209008240243104215933593992;

}  // namespace probability_constant

/*! \brief Enum encoding possible choices of probability distribution */
enum class ProbabilityDistributionType : int {
  kNormal = 0, kLogistic = 1, kExtreme = 2
};

/*! \brief Interface for a probability distribution */
class ProbabilityDistribution {
 public:
  /*!
   * \brief Evaluate Probability Density Function (PDF) at a particular point
   * \param z point at which to evaluate PDF
   * \return Value of PDF evaluated
   */
  virtual double PDF(double z) = 0;
  /*!
   * \brief Evaluate Cumulative Distribution Function (CDF) at a particular point
   * \param z point at which to evaluate CDF
   * \return Value of CDF evaluated
   */
  virtual double CDF(double z) = 0;
  /*!
   * \brief Evaluate first derivative of PDF at a particular point
   * \param z point at which to evaluate first derivative of PDF
   * \return Value of first derivative of PDF evaluated
   */
  virtual double GradPDF(double z) = 0;
  /*!
   * \brief Evaluate second derivative of PDF at a particular point
   * \param z point at which to evaluate second derivative of PDF
   * \return Value of second derivative of PDF evaluated
   */
  virtual double HessPDF(double z) = 0;

  /*!
   * \brief Factory function to instantiate a new probability distribution object
   * \param dist kind of probability distribution
   * \return Reference to the newly created probability distribution object
   */
  static ProbabilityDistribution* Create(ProbabilityDistributionType dist);
  virtual ~ProbabilityDistribution() = default;
};

/*! \brief The (standard) normal distribution */
class NormalDist : public ProbabilityDistribution {
 public:
  double PDF(double z) override;
  double CDF(double z) override;
  double GradPDF(double z) override;
  double HessPDF(double z) override;
};

/*! \brief The (standard) logistic distribution */
class LogisticDist : public ProbabilityDistribution {
 public:
  double PDF(double z) override;
  double CDF(double z) override;
  double GradPDF(double z) override;
  double HessPDF(double z) override;
};

/*! \brief The extreme distribution, also known as the Gumbel (minimum) distribution */
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
