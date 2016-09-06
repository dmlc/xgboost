/*!
 *  Copyright (c) 2015 by Contributors
 * \file recordio.h
 * \brief recordio that is able to pack binary data into a splittable
 *   format, useful to exchange data in binary serialization,
 *   such as binary raw data or protobuf
 */
#ifndef DMLC_RECORDIO_H_
#define DMLC_RECORDIO_H_
#include <cstring>
#include <string>
#include "./io.h"
#include "./logging.h"

namespace dmlc {
/*!
 * \brief writer of binary recordio
 *  binary format for recordio
 *  recordio format: magic lrecord data pad
 *
 *  - magic is magic number
 *  - pad is simply a padding space to make record align to 4 bytes
 *  - lrecord encodes length and continue bit
 *     - data.length() = (lrecord & (1U<<29U - 1));
 *     - cflag == (lrecord >> 29U) & 7;
 *
 *  cflag was used to handle (rare) special case when magic number
 *  occured in the data sequence.
 *
 *  In such case, the data is splitted into multiple records by
 *  the cells of magic number
 *
 *  (1) cflag == 0: this is a complete record;
 *  (2) cflag == 1: start of a multiple-rec;
 *      cflag == 2: middle of multiple-rec;
 *      cflag == 3: end of multiple-rec
 */
class RecordIOWriter {
 public:
  /*!
   * \brief magic number of recordio
   * note: (kMagic >> 29U) & 7 > 3
   * this ensures lrec will not be kMagic
   */
  static const uint32_t kMagic = 0xced7230a;
  /*!
   * \brief encode the lrecord
   * \param cflag cflag part of the lrecord
   * \param length length part of lrecord
   * \return the encoded data
   */
  inline static uint32_t EncodeLRec(uint32_t cflag, uint32_t length) {
    return (cflag << 29U) | length;
  }
  /*!
   * \brief decode the flag part of lrecord
   * \param rec the lrecord
   * \return the flag
   */
  inline static uint32_t DecodeFlag(uint32_t rec) {
    return (rec >> 29U) & 7U;
  }
  /*!
   * \brief decode the length part of lrecord
   * \param rec the lrecord
   * \return the length
   */
  inline static uint32_t DecodeLength(uint32_t rec) {
    return rec & ((1U << 29U) - 1U);
  }
  /*!
   * \brief constructor
   * \param stream the stream to be constructed
   */
  explicit RecordIOWriter(Stream *stream)
      : stream_(stream), seek_stream_(dynamic_cast<SeekStream*>(stream)),
        except_counter_(0) {
    CHECK(sizeof(uint32_t) == 4) << "uint32_t needs to be 4 bytes";
  }
  /*!
   * \brief write record to the stream
   * \param buf the buffer of memory region
   * \param size the size of record to write out
   */
  void WriteRecord(const void *buf, size_t size);
  /*!
   * \brief write record to the stream
   * \param data the data to write out
   */
  inline void WriteRecord(const std::string &data) {
    this->WriteRecord(data.c_str(), data.length());
  }
  /*!
   * \return number of exceptions(occurance of magic number)
   *   during the writing process
   */
  inline size_t except_counter(void) const {
    return except_counter_;
  }

  /*! \brief tell the current position of the input stream */
  inline size_t Tell(void) {
    CHECK(seek_stream_ != NULL) << "The input stream is not seekable";
    return seek_stream_->Tell();
  }

 private:
  /*! \brief output stream */
  Stream *stream_;
  /*! \brief seekable stream */
  SeekStream *seek_stream_;
  /*! \brief counts the number of exceptions */
  size_t except_counter_;
};
/*!
 * \brief reader of binary recordio to reads in record from stream
 * \sa RecordIOWriter
 */
class RecordIOReader {
 public:
  /*!
   * \brief constructor
   * \param stream the stream to be constructed
   */
  explicit RecordIOReader(Stream *stream)
      : stream_(stream), seek_stream_(dynamic_cast<SeekStream*>(stream)),
        end_of_stream_(false) {
    CHECK(sizeof(uint32_t) == 4) << "uint32_t needs to be 4 bytes";
  }
  /*!
   * \brief read next complete record from stream
   * \param out_rec used to store output record in string
   * \return true of read was successful, false if end of stream was reached
   */
  bool NextRecord(std::string *out_rec);

  /*! \brief seek to certain position of the input stream */
  inline void Seek(size_t pos) {
    CHECK(seek_stream_ != NULL) << "The input stream is not seekable";
    seek_stream_->Seek(pos);
  }

 private:
  /*! \brief output stream */
  Stream *stream_;
  SeekStream *seek_stream_;
  /*! \brief whether we are at end of stream */
  bool end_of_stream_;
};

/*!
 * \brief reader of binary recordio from Blob returned by InputSplit
 *  This class divides the blob into several independent parts specified by caller,
 *  and read from one segment.
 *  The part reading can be used together with InputSplit::NextChunk for
 *  multi-threaded parsing(each thread take a RecordIOChunkReader)
 *
 * \sa RecordIOWriter, InputSplit
 */
class RecordIOChunkReader {
 public:
  /*!
   * \brief constructor
   * \param chunk source data returned by InputSplit
   * \param part_index which part we want to reado
   * \param num_parts number of total segments
   */
  explicit RecordIOChunkReader(InputSplit::Blob chunk,
                               unsigned part_index = 0,
                               unsigned num_parts = 1);
  /*!
   * \brief read next complete record from stream
   *   the blob contains the memory content
   *   NOTE: this function is not threadsafe, use one
   *   RecordIOChunkReader per thread
   * \param out_rec used to store output blob, the header is already
   *        removed and out_rec only contains the memory content
   * \return true of read was successful, false if end was reached
   */
  bool NextRecord(InputSplit::Blob *out_rec);

 private:
  /*! \brief internal temporal data */
  std::string temp_;
  /*! \brief internal data pointer */
  char *pbegin_, *pend_;
};

}  // namespace dmlc
#endif  // DMLC_RECORDIO_H_
