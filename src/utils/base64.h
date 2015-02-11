#ifndef XGBOOST_UTILS_BASE64_H_
#define XGBOOST_UTILS_BASE64_H_
/*!
 * \file base64.h
 * \brief data stream support to input and output from/to base64 stream
 * base64 is easier to store and pass as text format in mapreduce
 * \author Tianqi Chen
 */
#include <cctype>
#include <cstdio>
#include "./utils.h"
#include "./io.h"

namespace xgboost {
namespace utils {
/*! \brief namespace of base64 decoding and encoding table */
namespace base64 {
const char DecodeTable[] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  62,  // '+'
  0, 0, 0,
  63,  // '/'
  52, 53, 54, 55, 56, 57, 58, 59, 60, 61,  // '0'-'9'
  0, 0, 0, 0, 0, 0, 0,
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,  // 'A'-'Z'
  0, 0, 0, 0, 0, 0,
  26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
  39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,  // 'a'-'z'
};
static const char EncodeTable[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
} // namespace base64
/*! \brief the stream that reads from base64, note we take from file pointers */
class Base64InStream: public IStream {
 public:
  explicit Base64InStream(FILE *fp) : fp(fp) {
    num_prev = 0; tmp_ch = 0;
  }
  /*! 
   * \brief initialize the stream position to beginning of next base64 stream 
   * call this function before actually start read
   */
  inline void InitPosition(void) {
    // get a charater
    do {
      tmp_ch = fgetc(fp);
    } while (isspace(tmp_ch));
  }
  /*! \brief whether current position is end of a base64 stream */
  inline bool IsEOF(void) const {
    return num_prev == 0 && (tmp_ch == EOF || isspace(tmp_ch));
  }
  virtual size_t Read(void *ptr, size_t size) {
    using base64::DecodeTable;
    if (size == 0) return 0;
    // use tlen to record left size
    size_t tlen = size;
    unsigned char *cptr = static_cast<unsigned char*>(ptr);
    // if anything left, load from previous buffered result
    if (num_prev != 0) {
      if (num_prev == 2) {
        if (tlen >= 2) {
          *cptr++ = buf_prev[0];
          *cptr++ = buf_prev[1];
          tlen -= 2;
          num_prev = 0;
        } else {
          // assert tlen == 1
          *cptr++ = buf_prev[0]; --tlen;
          buf_prev[0] = buf_prev[1];
          num_prev = 1;
        }
      } else {
        // assert num_prev == 1
        *cptr++ = buf_prev[0]; --tlen; num_prev = 0;
      }
    }
    if (tlen == 0) return size;
    int nvalue;
    // note: everything goes with 4 bytes in Base64
    // so we process 4 bytes a unit
    while (tlen && tmp_ch != EOF && !isspace(tmp_ch)) {
      // first byte
      nvalue = DecodeTable[tmp_ch] << 18;
      {
        // second byte
        Check((tmp_ch = fgetc(fp), tmp_ch != EOF && !isspace(tmp_ch)),
              "invalid base64 format");
        nvalue |= DecodeTable[tmp_ch] << 12;
        *cptr++ = (nvalue >> 16) & 0xFF; --tlen;
      }
      {
        // third byte
        Check((tmp_ch = fgetc(fp), tmp_ch != EOF && !isspace(tmp_ch)),
              "invalid base64 format");
        // handle termination
        if (tmp_ch == '=') {
          Check((tmp_ch = fgetc(fp), tmp_ch == '='), "invalid base64 format");
          Check((tmp_ch = fgetc(fp), tmp_ch == EOF || isspace(tmp_ch)),
                "invalid base64 format");
          break;
        }
        nvalue |= DecodeTable[tmp_ch] << 6;
        if (tlen) {
          *cptr++ = (nvalue >> 8) & 0xFF; --tlen;
        } else {
          buf_prev[num_prev++] = (nvalue >> 8) & 0xFF;
        }
      }
      {
        // fourth byte
        Check((tmp_ch = fgetc(fp), tmp_ch != EOF && !isspace(tmp_ch)),
              "invalid base64 format");
        if (tmp_ch == '=') {
          Check((tmp_ch = fgetc(fp), tmp_ch == EOF || isspace(tmp_ch)),
                "invalid base64 format");
          break;
        }
        nvalue |= DecodeTable[tmp_ch];
        if (tlen) {
          *cptr++ = nvalue & 0xFF; --tlen;
        } else {
          buf_prev[num_prev ++] = nvalue & 0xFF;
        }
      }
      // get next char
      tmp_ch = fgetc(fp);
    }
    if (kStrictCheck) {
      Check(tlen == 0, "Base64InStream: read incomplete");
    }
    return size - tlen;
  }
  virtual void Write(const void *ptr, size_t size) {
    utils::Error("Base64InStream do not support write");
  }

 private:
  FILE *fp;
  int tmp_ch;
  int num_prev;
  unsigned char buf_prev[2];
  // whether we need to do strict check
  static const bool kStrictCheck = false;
};
/*! \brief the stream that write to base64, note we take from file pointers */
class Base64OutStream: public IStream {
 public:
  explicit Base64OutStream(FILE *fp) : fp(fp) {
    buf_top = 0;
  }
  virtual void Write(const void *ptr, size_t size) {
    using base64::EncodeTable;
    size_t tlen = size;
    const unsigned char *cptr = static_cast<const unsigned char*>(ptr);
    while (tlen) {
      while (buf_top < 3  && tlen != 0) {
        buf[++buf_top] = *cptr++; --tlen;
      }
      if (buf_top == 3) {
        // flush 4 bytes out
        fputc(EncodeTable[buf[1] >> 2], fp);
        fputc(EncodeTable[((buf[1] << 4) | (buf[2] >> 4)) & 0x3F], fp);
        fputc(EncodeTable[((buf[2] << 2) | (buf[3] >> 6)) & 0x3F], fp);
        fputc(EncodeTable[buf[3] & 0x3F], fp);
        buf_top = 0;
      }
    }
  }
  virtual size_t Read(void *ptr, size_t size) {
    Error("Base64OutStream do not support read");
    return 0;
  }
  /*!
   * \brief finish writing of all current base64 stream, do some post processing
   * \param endch charater to put to end of stream, if it is EOF, then nothing will be done
   */
  inline void Finish(char endch = EOF) {
    using base64::EncodeTable;
    if (buf_top == 1) {
      fputc(EncodeTable[buf[1] >> 2], fp);
      fputc(EncodeTable[(buf[1] << 4) & 0x3F], fp);
      fputc('=', fp);
      fputc('=', fp);
    }
    if (buf_top == 2) {
      fputc(EncodeTable[buf[1] >> 2], fp);
      fputc(EncodeTable[((buf[1] << 4) | (buf[2] >> 4)) & 0x3F], fp);
      fputc(EncodeTable[(buf[2] << 2) & 0x3F], fp);
      fputc('=', fp);
    }
    buf_top = 0;
    if (endch != EOF) fputc(endch, fp);
  }

 private:
  FILE *fp;
  int buf_top;
  unsigned char buf[4];
};
}  // namespace utils
}  // namespace xgboost
#endif  // XGBOOST_UTILS_BASE64_H_
