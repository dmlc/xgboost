#ifndef RABIT_LEARN_IO_IO_INL_H_
#define RABIT_LEARN_IO_IO_INL_H_
/*!
 * \file io-inl.h
 * \brief Input/Output utils that handles read/write
 *        of files in distrubuted enviroment
 * \author Tianqi Chen
 */
#include <cstring>

#include "./io.h"

#if RABIT_USE_WORMHOLE == 0
#if RABIT_USE_HDFS
#include "./hdfs-inl.h"
#endif
#include "./file-inl.h"
#endif

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
#if RABIT_USE_WORMHOLE
  return dmlc::InputSplit::Create(uri, part, nsplit);
#else
  using namespace std;
  if (!strcmp(uri, "stdin")) {
    return new SingleFileSplit(uri);
  }
  if (!strncmp(uri, "file://", 7)) {
    return new LineSplitter(new FileProvider(uri), part, nsplit);
  }
  if (!strncmp(uri, "hdfs://", 7)) {
#if RABIT_USE_HDFS
    return new LineSplitter(new HDFSProvider(uri), part, nsplit);
#else
    utils::Error("Please compile with RABIT_USE_HDFS=1");
#endif
  }
  return new LineSplitter(new FileProvider(uri), part, nsplit);
#endif
}

template<typename TStream>
class StreamAdapter : public Stream {
 public:
  explicit StreamAdapter(TStream *stream)
      : stream_(stream) {
  }
  virtual ~StreamAdapter(void) {
    delete stream_;
  }
  virtual size_t Read(void *ptr, size_t size) {
    return stream_->Read(ptr, size);
  }
  virtual void Write(const void *ptr, size_t size) {
    stream_->Write(ptr, size);
  }  
 private:
  TStream *stream_;
};

/*!
 * \brief create an stream, the stream must be able to close
 *    the underlying resources(files) when deleted
 *
 * \param uri the uri of the input, can contain hdfs prefix
 * \param mode can be 'w' or 'r' for read or write
 */
inline Stream *CreateStream(const char *uri, const char *mode) {
#if RABIT_USE_WORMHOLE
  return new StreamAdapter<dmlc::Stream>(dmlc::Stream::Create(uri, mode));
#else
  using namespace std;
  if (!strncmp(uri, "file://", 7)) {
    return new FileStream(uri + 7, mode);
  }
  if (!strncmp(uri, "hdfs://", 7)) {
#if RABIT_USE_HDFS
    return new HDFSStream(hdfsConnect(HDFSStream::GetNameNode().c_str(), 0),
                          uri, mode, true);
#else
    utils::Error("Please compile with RABIT_USE_HDFS=1");
#endif
  }
  return new FileStream(uri, mode);
#endif
}
}  // namespace io
}  // namespace rabit
#endif  // RABIT_LEARN_IO_IO_INL_H_
