// Copyright by Contributors

#include <dmlc/base.h>
#include <dmlc/recordio.h>
#include <dmlc/logging.h>
#include <algorithm>


namespace dmlc {
// implemmentation
void RecordIOWriter::WriteRecord(const void *buf, size_t size) {
  CHECK(size < (1 << 29U))
      << "RecordIO only accept record less than 2^29 bytes";
  const uint32_t umagic = kMagic;
  // initialize the magic number, in stack
  const char *magic = reinterpret_cast<const char*>(&umagic);
  const char *bhead = reinterpret_cast<const char*>(buf);
  uint32_t len = static_cast<uint32_t>(size);
  uint32_t lower_align = (len >> 2U) << 2U;
  uint32_t upper_align = ((len + 3U) >> 2U) << 2U;
  uint32_t dptr = 0;
  for (uint32_t i = 0; i < lower_align ; i += 4) {
    // use char check for alignment safety reason
    if (bhead[i] == magic[0] &&
        bhead[i + 1] == magic[1] &&
        bhead[i + 2] == magic[2] &&
        bhead[i + 3] == magic[3]) {
      uint32_t lrec = EncodeLRec(dptr == 0 ? 1U : 2U,
                                 i - dptr);
      stream_->Write(magic, 4);
      stream_->Write(&lrec, sizeof(lrec));
      if (i != dptr) {
        stream_->Write(bhead + dptr, i - dptr);
      }
      dptr = i + 4;
      except_counter_ += 1;
    }
  }
  uint32_t lrec = EncodeLRec(dptr != 0 ? 3U : 0U,
                             len - dptr);
  stream_->Write(magic, 4);
  stream_->Write(&lrec, sizeof(lrec));
  if (len != dptr) {
    stream_->Write(bhead + dptr, len - dptr);
  }
  // write padded bytes
  uint32_t zero = 0;
  if (upper_align != len) {
    stream_->Write(&zero, upper_align - len);
  }
}

bool RecordIOReader::NextRecord(std::string *out_rec) {
  if (end_of_stream_) return false;
  const uint32_t kMagic = RecordIOWriter::kMagic;
  out_rec->clear();
  size_t size = 0;
  while (true) {
    uint32_t header[2];
    size_t nread = stream_->Read(header, sizeof(header));
    if (nread == 0) {
      end_of_stream_ = true; return false;
    }
    CHECK(nread == sizeof(header)) << "Inavlid RecordIO File";
    CHECK(header[0] == RecordIOWriter::kMagic) << "Invalid RecordIO File";
    uint32_t cflag = RecordIOWriter::DecodeFlag(header[1]);
    uint32_t len = RecordIOWriter::DecodeLength(header[1]);
    uint32_t upper_align = ((len + 3U) >> 2U) << 2U;
    out_rec->resize(size + upper_align);
    if (upper_align != 0) {
      CHECK(stream_->Read(BeginPtr(*out_rec) + size, upper_align) == upper_align)
          << "Invalid RecordIO File upper_align=" << upper_align;
    }
    // squeeze back
    size += len; out_rec->resize(size);
    if (cflag == 0U || cflag == 3U) break;
    out_rec->resize(size + sizeof(kMagic));
    std::memcpy(BeginPtr(*out_rec) + size, &kMagic, sizeof(kMagic));
    size += sizeof(kMagic);
  }
  return true;
}

// helper function to find next recordio head
inline char *FindNextRecordIOHead(char *begin, char *end) {
  CHECK_EQ((reinterpret_cast<size_t>(begin) & 3UL),  0);
  CHECK_EQ((reinterpret_cast<size_t>(end) & 3UL), 0);
  uint32_t *p = reinterpret_cast<uint32_t *>(begin);
  uint32_t *pend = reinterpret_cast<uint32_t *>(end);
  for (; p + 1 < pend; ++p) {
    if (p[0] == RecordIOWriter::kMagic) {
      uint32_t cflag = RecordIOWriter::DecodeFlag(p[1]);
      if (cflag == 0 || cflag == 1) {
        return reinterpret_cast<char*>(p);
      }
    }
  }
  return end;
}

RecordIOChunkReader::RecordIOChunkReader(InputSplit::Blob chunk,
                                         unsigned part_index,
                                         unsigned num_parts) {
  size_t nstep = (chunk.size + num_parts - 1) / num_parts;
  // align
  nstep = ((nstep + 3UL) >> 2UL) << 2UL;
  size_t begin = std::min(chunk.size, nstep * part_index);
  size_t end = std::min(chunk.size, nstep * (part_index + 1));
  char *head = reinterpret_cast<char*>(chunk.dptr);
  pbegin_ = FindNextRecordIOHead(head + begin, head + chunk.size);
  pend_ = FindNextRecordIOHead(head + end, head + chunk.size);
}

bool RecordIOChunkReader::NextRecord(InputSplit::Blob *out_rec) {
  if (pbegin_ >= pend_) return false;
  uint32_t *p = reinterpret_cast<uint32_t *>(pbegin_);
  CHECK(p[0] == RecordIOWriter::kMagic);
  uint32_t cflag = RecordIOWriter::DecodeFlag(p[1]);
  uint32_t clen = RecordIOWriter::DecodeLength(p[1]);
  if (cflag == 0) {
    // skip header
    out_rec->dptr = pbegin_ + 2 * sizeof(uint32_t);
    // move pbegin
    pbegin_ += 2 * sizeof(uint32_t) + (((clen + 3U) >> 2U) << 2U);
    CHECK(pbegin_ <= pend_) << "Invalid RecordIO Format";
    out_rec->size = clen;
    return true;
  } else {
    const uint32_t kMagic = RecordIOWriter::kMagic;
    // abnormal path, read into string
    CHECK(cflag == 1U) << "Invalid RecordIO Format";
    temp_.resize(0);
    while (true) {
      CHECK(pbegin_ + 2 * sizeof(uint32_t) <= pend_);
      p = reinterpret_cast<uint32_t *>(pbegin_);
      CHECK(p[0] == RecordIOWriter::kMagic);
      cflag = RecordIOWriter::DecodeFlag(p[1]);
      clen = RecordIOWriter::DecodeLength(p[1]);
      size_t tsize = temp_.length();
      temp_.resize(tsize + clen);
      if (clen != 0) {
        std::memcpy(BeginPtr(temp_) + tsize,
                    pbegin_ + 2 * sizeof(uint32_t),
                    clen);
        tsize += clen;
      }
      pbegin_ += 2 * sizeof(uint32_t) + (((clen + 3U) >> 2U) << 2U);
      if (cflag == 3U) break;
      temp_.resize(tsize + sizeof(kMagic));
      std::memcpy(BeginPtr(temp_) + tsize, &kMagic, sizeof(kMagic));
    }
    out_rec->dptr = BeginPtr(temp_);
    out_rec->size = temp_.length();
    return true;
  }
}
}  // namespace dmlc
