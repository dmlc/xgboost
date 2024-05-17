/**
 * Copyright 2014-2024 by XGBoost Contributors
 */

#include "processing/heu_processor.h"

#include <fcntl.h>

#include "absl/strings/numbers.h"
#include "heu/library/phe/encoding/batch_encoder.h"

namespace processing {

namespace {

using heu::lib::numpy::DestinationHeKit;
using heu::lib::numpy::HeKit;

using heu::lib::numpy::CMatrix;
using heu::lib::numpy::PMatrix;

using heu::lib::phe::BatchEncoder;
using heu::lib::phe::Ciphertext;
using heu::lib::phe::Plaintext;

using std::string;

const char kPublicKeyPath[] = "PUBLIC_KEY_PATH";
const char kSecretKeyPath[] = "SECRET_KEY_PATH";
const char pk_file[] = "public-key";
const char sk_file[] = "secret-key";
const int64_t kDefaultScale = 1e6;

yacl::Buffer ReadFile(std::string_view file_name) {
  int fd = open(file_name.data(), O_RDONLY);
  YACL_ENFORCE(fd != -1, "errno {}, {}", errno, strerror(errno));

  yacl::Buffer buf;
  const int cnt = 100;
  buf.reserve(cnt);
  ssize_t num_read;
  while ((num_read = read(fd, buf.data<std::byte>() + buf.size(), cnt)) > 0) {
    YACL_ENFORCE(num_read != -1, "errno {}, {}", errno, strerror(errno));
    buf.resize(buf.size() + num_read);
    buf.reserve(buf.size() + cnt);
  }

  close(fd);
  return buf;
}

yacl::Buffer GetPublicKey(const std::map<string, string> &params) {
  auto it = params.find(kPublicKeyPath);
  if (it == params.end()) {
    return ReadFile(pk_file);
  } else {
    return ReadFile(absl::StrCat(it->second, "/", pk_file));
  }
}

yacl::Buffer GetSecretKey(const std::map<string, string> &params) {
  auto it = params.find(kSecretKeyPath);
  if (it == params.end()) {
    return ReadFile(sk_file);
  } else {
    return ReadFile(absl::StrCat(it->second, "/", sk_file));
  }
}

int64_t GetScale(const std::map<string, string> &params) {
  auto it = params.find("scale");
  if (it == params.end()) {
    return kDefaultScale;
  }

  int64_t scale;
  YACL_ENFORCE(absl::SimpleAtoi(it->second, &scale));
  YACL_ENFORCE(scale > 0);
  return scale;
}

}  // namespace

void HeuProcessor::Initialize(bool active, std::map<string, string> params) {
  active_ = active;

  auto pk_buffer = GetPublicKey(params);
  if (active_) {
    auto sk_buffer = GetSecretKey(params);
    he_kit_ =
        std::make_unique<HeKit>(heu::lib::phe::HeKit(pk_buffer, sk_buffer));
    scale_ = GetScale(params);
  } else {
    dest_he_kit_ = std::make_unique<DestinationHeKit>(
        heu::lib::phe::DestinationHeKit(pk_buffer));
  }
}

void HeuProcessor::Shutdown() {
  this->cuts_.clear();
  this->slots_.clear();

  he_kit_ = nullptr;
  dest_he_kit_ = nullptr;
  gh_ = nullptr;
  scale_ = 0;
}

void HeuProcessor::FreeBuffer(void *buffer) { free(buffer); }

void *HeuProcessor::ProcessGHPairs(size_t *size,
                                   const std::vector<double> &pairs) {
  YACL_ENFORCE(active_, "only active party allowed to call this function");
  YACL_ENFORCE(he_kit_, "he_kit equals to nullptr");
  YACL_ENFORCE(scale_ > 0, "scale not set");

  auto encoder = he_kit_->GetEncoder<BatchEncoder>(scale_);
  PMatrix gh(pairs.size() / 2);

  gh.ForEach([&](int64_t row, int64_t, Plaintext *pt) {
    *pt = encoder.Encode(pairs[2 * row], pairs[2 * row + 1]);
  });

  auto encryptor = he_kit_->GetEncryptor();
  gh_ = std::make_unique<CMatrix>(encryptor->Encrypt(gh));
  auto buf = gh_->Serialize();
  *size = buf.size();
  return buf.release();
}

void *HeuProcessor::HandleGHPairs(size_t *size, void *buffer, size_t buf_size) {
  *size = buf_size;
  gh_ = std::make_unique<CMatrix>(
      CMatrix::LoadFrom(yacl::ByteContainerView(buffer, buf_size)));

  return buffer;  // TODO: directly return buffer?
}

void HeuProcessor::InitAggregationContext(const std::vector<uint32_t> &cuts,
                                          const std::vector<int> &slots) {
  this->cuts_ = cuts;
  if (this->slots_.empty()) {
    this->slots_ = slots;
  }
}

void *HeuProcessor::ProcessAggregation(size_t *size,
                                       std::map<int, std::vector<int>> nodes) {
  YACL_ENFORCE(dest_he_kit_, "dest_he_kit equals to nullptr");
  YACL_ENFORCE(gh_, "GH ciphertext matrix not set");

  auto evaluator = dest_he_kit_->GetEvaluator();
  auto encryptor = dest_he_kit_->GetEncryptor();
  int total_bin_size = cuts_.back();
  auto feature_num = cuts_.size() - 1;

  CMatrix histograms(nodes.size(), total_bin_size);
  auto zero = encryptor->EncryptZero();
  histograms.ForEach([&](int64_t, int64_t, Ciphertext *pt) { *pt = zero; });

  int histo_i = 0;
  for (const auto &node : nodes) {
    const auto &rows = node.second;
    for (int row_id : rows) {
      yacl::parallel_for(0, feature_num, 1, [&](int64_t beg, int64_t end) {
        for (int64_t f = beg; f < end; ++f) {
          int slot = slots_[f + feature_num * row_id];
          if ((slot < 0) || (slot >= total_bin_size)) {
            continue;
          }
          const auto &gh = (*gh_)(row_id);
          evaluator->AddInplace(&histograms(histo_i, slot), gh);
        }
      });
    }
    ++histo_i;
  }

  auto buf = histograms.Serialize();
  *size = buf.size();
  return buf.release();
}

std::vector<double> HeuProcessor::HandleAggregation(void *buffer,
                                                    size_t buf_size) {
  YACL_ENFORCE(active_, "only active party allowed to call this function");
  YACL_ENFORCE(he_kit_, "he_kit equals to nullptr");
  YACL_ENFORCE(scale_ > 0, "scale not set");

  auto decryptor = he_kit_->GetDecryptor();
  auto encoder = he_kit_->GetEncoder<BatchEncoder>(scale_);
  size_t offset = 0;
  std::vector<double> result;

  while (offset != buf_size) {
    auto histogram = CMatrix::LoadFrom(
        yacl::ByteContainerView(buffer, buf_size),
        heu::lib::numpy::MatrixSerializeFormat::Best, &offset);
    auto plaintexts = decryptor->Decrypt(histogram);
    for (int i = 0; i < plaintexts.rows(); ++i) {
      for (int j = 0; j < plaintexts.cols(); ++j) {
        result.push_back(encoder.Decode<double, 0>(plaintexts(i, j)));
        result.push_back(encoder.Decode<double, 1>(plaintexts(i, j)));
      }
    }
  }

  return result;
}

extern "C" {
Processor *LoadProcessor(const char *) {
  return new processing::HeuProcessor();  // TODO: on heap?
}
}

}  // namespace processing
