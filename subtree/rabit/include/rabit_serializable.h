/*!
 *  Copyright (c) 2014 by Contributors
 * \file rabit_serializable.h
 * \brief defines serializable interface of rabit
 * \author Tianqi Chen
 */
#ifndef RABIT_RABIT_SERIALIZABLE_H_
#define RABIT_RABIT_SERIALIZABLE_H_
#include <vector>
#include <string>
#include "./rabit/utils.h"
#include "./dmlc/io.h"

namespace rabit {
/*!
 * \brief defines stream used in rabit 
 * see definition of IStream in dmlc/io.h 
 */
typedef dmlc::IStream IStream;
/*!
 * \brief defines serializable objects used in rabit 
 * see definition of ISerializable in dmlc/io.h 
 */
typedef dmlc::ISerializable ISerializable;

}  // namespace rabit
#endif  // RABIT_RABIT_SERIALIZABLE_H_
