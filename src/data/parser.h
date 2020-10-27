#ifndef XGBOOST_DATA_PARSER_H_
#define XGBOOST_DATA_PARSER_H_
#include <dmlc/io.h>
#include "../../dmlc-core/src/io/single_threaded_input_split.h"
#include "../../dmlc-core/src/io/single_file_split.h"
#include "../../dmlc-core/src/io/uri_spec.h"
#include "../../dmlc-core/src/io/line_split.h"
#include "../../dmlc-core/src/io/indexed_recordio_split.h"
#include "../../dmlc-core/src/io/recordio_split.h"
#include "../../dmlc-core/src/io/cached_input_split.h"
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
  bool NextChunk(Blob *out_chunk) override {
    if (tmp_chunk_ == nullptr) {
    }
    LOG(FATAL) << "Not implemented";
    return true;
  }

  void ResetPartition(unsigned part_index, unsigned num_parts) override {
    LOG(FATAL) << "Not implemented";
    base_->ResetPartition(part_index, num_parts);
    this->BeforeFirst();
  }
};

// template <typename IndexType, typename DType>
// class CSVParser : public dmlc::data::CSVParser<IndexType, DType> {
//   size_t bytes_read_ { 0 };
//   dmlc::InputSplit *source_;

//   bool ParseNext(std::vector<dmlc::data::RowBlockContainer<IndexType, DType> > *data) override {
//     dmlc::InputSplit::Blob chunk;
//     if (!source_->NextChunk(&chunk)) return false;
//     bytes_read_ += chunk.size;
//     CHECK_NE(chunk.size, 0U);
//     const char *head = reinterpret_cast<char *>(chunk.dptr);
//   }
// };

inline dmlc::InputSplit *
CreateInputSplit(const char *uri_, const char *index_uri_, unsigned part,
                 unsigned nsplit, const char *type, const bool shuffle = false,
                 const int seed = 0, const size_t batch_size = 256,
                 const bool recurse_directories = false) {
  namespace io = dmlc::io;
  io::URISpec spec(uri_, part, nsplit);
  if (!strcmp(spec.uri.c_str(), "stdin")) {
    return new io::SingleFileSplit(spec.uri.c_str());
  }
  CHECK(part < nsplit) << "invalid input parameter for InputSplit::Create";
  io::URI path(spec.uri.c_str());
  std::unique_ptr<io::InputSplitBase> split{nullptr};
  if (!strcmp(type, "text")) {
    split.reset(new io::LineSplitter(io::FileSystem::GetInstance(path),
                                     spec.uri.c_str(), part, nsplit));
  } else if (!strcmp(type, "indexed_recordio")) {
    if (index_uri_ != nullptr) {
      io::URISpec index_spec(index_uri_, part, nsplit);
      split.reset(new io::IndexedRecordIOSplitter(
          io::FileSystem::GetInstance(path), spec.uri.c_str(),
          index_spec.uri.c_str(), part, nsplit, batch_size, shuffle, seed));
    } else {
      LOG(FATAL) << "need to pass index file to use IndexedRecordIO";
    }
  } else if (!strcmp(type, "recordio")) {
    split.reset(new io::RecordIOSplitter(io::FileSystem::GetInstance(path),
                                         spec.uri.c_str(), part, nsplit,
                                         recurse_directories));
  } else {
    LOG(FATAL) << "unknown input split type " << type;
  }
  if (spec.cache_file.length() == 0) {
    return new TextInputSplit(std::move(split), batch_size);
  } else {
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
