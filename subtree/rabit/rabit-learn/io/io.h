#ifndef RABIT_LEARN_IO_IO_H_
#define RABIT_LEARN_IO_IO_H_
/*!
 * \file io.h
 * \brief Input/Output utils that handles read/write
 *        of files in distrubuted enviroment
 * \author Tianqi Chen
 */
#include "../../include/rabit_serializable.h"

/*! \brief whether compile with HDFS support */
#ifndef RABIT_USE_HDFS
#define RABIT_USE_HDFS 0
#endif

#ifndef RABIT_USE_WORMHOLE
#define RABIT_USE_WORMHOLE 0
#endif

#if RABIT_USE_WORMHOLE
#include <dmlc/io.h>
#endif
/*! \brief io interface */
namespace rabit {
/*!
 * \brief namespace to handle input split and filesystem interfacing
 */
namespace io {
/*! \brief reused ISeekStream's definition */
#if RABIT_USE_WORMHOLE
typedef dmlc::ISeekStream ISeekStream;
typedef dmlc::InputSplit InputSplit;
#else
typedef utils::ISeekStream ISeekStream;
/*!
 * \brief user facing input split helper,
 *   can be used to get the partition of data used by current node
 */
class InputSplit {
 public:
  /*!
   * \brief get next line, store into out_data
   * \param out_data the string that stores the line data,
   *        \n is not included
   * \return true of next line was found, false if we read all the lines
   */
  virtual bool ReadLine(std::string *out_data) = 0;
  /*! \brief destructor*/
  virtual ~InputSplit(void) {}
};
#endif
/*!
 * \brief create input split given a uri
 * \param uri the uri of the input, can contain hdfs prefix
 * \param part the part id of current input
 * \param nsplit total number of splits
 */
inline InputSplit *CreateInputSplit(const char *uri,
                                    unsigned part,
                                    unsigned nsplit);
/*!
 * \brief create an stream, the stream must be able to close
 *    the underlying resources(files) when deleted
 *
 * \param uri the uri of the input, can contain hdfs prefix
 * \param mode can be 'w' or 'r' for read or write
 */
inline IStream *CreateStream(const char *uri, const char *mode);
}  // namespace io
}  // namespace rabit

#include "./io-inl.h"
#include "./base64-inl.h"
#endif  // RABIT_LEARN_IO_IO_H_
