#ifndef XGBOOST_DATA_PARSER_H_
#define XGBOOST_DATA_PARSER_H_
#include <dmlc/io.h>
#include "../../dmlc-core/src/io/uri_spec.h"
#include "../../dmlc-core/src/io/line_split.h"
#include "../../dmlc-core/src/data/csv_parser.h"
#include "../../dmlc-core/src/data/libfm_parser.h"
#include "../../dmlc-core/src/data/libsvm_parser.h"

#include "xgboost/logging.h"

namespace xgboost {
namespace data {

class TextInputSplit : public dmlc::InputSplit {
  dmlc::io::InputSplitBase::Chunk *tmp_chunk_;
  std::unique_ptr<dmlc::io::InputSplitBase> base_;
  size_t batch_size_;

 public:
  TextInputSplit(std::unique_ptr<dmlc::io::InputSplitBase> &&split,
                 size_t batch_size)
      : base_{std::forward<std::unique_ptr<dmlc::io::InputSplitBase>>(split)},
        batch_size_{batch_size} {}

  size_t GetTotalSize() override {
    LOG(FATAL) << "Not implemented";
    return 0;
  }
  /*! \brief reset the position of InputSplit to beginning */
  void BeforeFirst() override {
    base_->BeforeFirst();
    if (tmp_chunk_ != nullptr) {
      tmp_chunk_ = nullptr;
    }
  }
  bool NextRecord(Blob *out_rec) override {
    LOG(FATAL) << "Not implemented";
    return false;
  }
  void ResetPartition(unsigned part_index, unsigned num_parts) override {
    LOG(FATAL) << "Not implemented";
  }

  virtual bool NextChunkEx(dmlc::io::InputSplitBase::Chunk *chunk) {
    if (!chunk->Load(base_.get(), batch_size_)) return false;
    return true;
  }

  bool NextChunk(Blob *out_chunk) override {
    if (tmp_chunk_ == nullptr) {
      tmp_chunk_ = new dmlc::io::InputSplitBase::Chunk(batch_size_);
    }

    out_chunk->dptr = tmp_chunk_->begin;
    out_chunk->size = tmp_chunk_->end - tmp_chunk_->begin;

    while (!base_->ExtractNextRecord(out_chunk, tmp_chunk_)) {
      if (!NextChunkEx(tmp_chunk_)) {
        return false;
      }
    }

    return true;
  }
};

inline dmlc::InputSplit *
CreateInputSplit(const char *uri_, const char *index_uri_, unsigned part,
                 unsigned nsplit, const char *type, const bool shuffle = false,
                 const int seed = 0, const size_t batch_size = 256,
                 const bool recurse_directories = false) {
  namespace io = dmlc::io;
  io::URISpec spec(uri_, part, nsplit);
  CHECK(part < nsplit) << "invalid input parameter for InputSplit::Create";
  io::URI path(spec.uri.c_str());
  std::unique_ptr<io::InputSplitBase> split{nullptr};
  if (!strcmp(type, "text")) {
    split.reset(new io::LineSplitter(io::FileSystem::GetInstance(path),
                                     spec.uri.c_str(), part, nsplit));
  } else {
    LOG(FATAL) << "unknown input split type " << type;
  }
  if (spec.cache_file.length() == 0) {
    return new TextInputSplit(std::move(split), batch_size);
  } else {
    // FIXME: This might be useful for distributed setting.
    LOG(FATAL) << "Not implemented";
    return nullptr;
  }
}

template<typename IndexType, typename DType = float>
dmlc::Parser<IndexType, DType> *
CreateCSVParser(const std::string& path,
                const std::map<std::string, std::string>& args,
                unsigned part_index,
                unsigned num_parts) {
  dmlc::InputSplit *source =
      CreateInputSplit(path.c_str(), nullptr, part_index, num_parts, "text");
  return new dmlc::data::CSVParser<IndexType, DType>(source, args, 2);
}

template<typename IndexType, typename DType = float>
dmlc::Parser<IndexType> *
CreateLibSVMParser(const std::string& path,
                   const std::map<std::string, std::string>& args,
                   unsigned part_index,
                   unsigned num_parts) {
  dmlc::InputSplit *source =
      CreateInputSplit(path.c_str(), nullptr, part_index, num_parts, "text");
  dmlc::data::ParserImpl<IndexType> *parser =
      new dmlc::data::LibSVMParser<IndexType>(source, args, 2);
  return parser;
}

template<typename IndexType, typename DType = float>
dmlc::Parser<IndexType> *
CreateLibFMParser(const std::string& path,
                  const std::map<std::string, std::string>& args,
                  unsigned part_index,
                  unsigned num_parts) {
  dmlc::InputSplit *source =
      CreateInputSplit(path.c_str(), nullptr, part_index, num_parts, "text");
  dmlc::data::ParserImpl<IndexType> *parser =
      new dmlc::data::LibFMParser<IndexType>(source, args, 2);
  return parser;
}

template <typename IndexType, typename DType = float>
inline dmlc::Parser<IndexType, DType> *
CreateParser(const char *uri_, unsigned part_index, unsigned num_parts,
             const char *type) {
  std::string ptype = type;
  dmlc::io::URISpec spec(uri_, part_index, num_parts);
  if (ptype == "auto") {
    if (spec.args.count("format") != 0) {
      ptype = spec.args.at("format");
    } else {
      ptype = "libsvm";
    }
  }

  // create parser
  if (ptype == "csv") {
    return CreateCSVParser<IndexType, DType>(spec.uri, spec.args, part_index, num_parts);
  } else {
    LOG(FATAL) << "Unknown file format: " << ptype;
  }
}
}  // namespace data
}  // namespace xgboost

#endif  // XGBOOST_DATA_PARSER_H_
