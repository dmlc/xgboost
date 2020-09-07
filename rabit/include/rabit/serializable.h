/*!
 *  Copyright (c) 2014 by Contributors
 * \file serializable.h
 * \brief defines serializable interface of rabit
 * \author Tianqi Chen
 */
#ifndef RABIT_SERIALIZABLE_H_
#define RABIT_SERIALIZABLE_H_
#include <vector>
#include <string>
#include "rabit/internal/utils.h"

namespace rabit {
/*!
 * \brief defines stream used in rabit
 * see definition of Stream in dmlc/io.h
 */
using Stream = dmlc::Stream ;
/*!
 * \brief defines serializable objects used in rabit
 * see definition of Serializable in dmlc/io.h
 */
using Serializable = dmlc::Serializable;

}  // namespace rabit
#endif  // RABIT_SERIALIZABLE_H_
