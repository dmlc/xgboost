/*!
 * Copyright 2014 by Contributors
 * \file io.h
 * \brief general stream interface for serialization, I/O
 * \author Tianqi Chen
 */

#ifndef XGBOOST_COMMON_IO_H_
#define XGBOOST_COMMON_IO_H_

#include <dmlc/io.h>
#include "./sync.h"

namespace xgboost {
namespace common {
typedef rabit::utils::MemoryFixSizeBuffer MemoryFixSizeBuffer;
typedef rabit::utils::MemoryBufferStream MemoryBufferStream;
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_IO_H_
