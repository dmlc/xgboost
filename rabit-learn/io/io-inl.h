#ifndef RABIT_LEARN_IO_IO_INL_H_
#define RABIT_LEARN_IO_IO_INL_H_
/*!
 * \file io-inl.h
 * \brief Input/Output utils that handles read/write
 *        of files in distrubuted enviroment
 * \author Tianqi Chen
 */
#include <cstring>
#include "./line_split.h"
namespace rabit {
namespace io {
/*!
 * \brief create input split given a uri
 * \param uri the uri of the input, can contain hdfs prefix
 * \param part the part id of current input
 * \param nsplit total number of splits
 */
inline InputSplit *CreateInputSplit(const char *uri,
                                    unsigned part,
                                    unsigned nsplit) {
  if (!strcmp(uri, "stdin")) {
    return new SingleFileSplit(uri);
  }
  if (!strncmp(uri, "file://", 7)) {
    return new FileSplit(uri, part, nsplit);
  }
  if (!strncmp(uri, "hdfs://", 7)) {
    utils::Error("HDFS reading is not yet supported");
    return NULL;
  }
  return new FileSplit(uri, part, nsplit);  
}
}  // namespace io
}  // namespace rabit
#endif  // RABIT_LEARN_IO_IO_INL_H_
