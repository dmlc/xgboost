/*!
 * Copyright 2019 XGBoost contributors
 */
#include <dmlc/io.h>

#include <string>
#include <tuple>
#include <vector>

#include "xgboost/logging.h"
#include "xgboost/version_config.h"
#include "version.h"
#include "json_experimental.h"

namespace xgboost {

const Version::TripletT Version::kInvalid {-1, -1, -1};

Version::TripletT Version::Load(experimental::Json const& in, bool check) {
  if (in.FindMemberByKey("version") == in.cend()) {
    return kInvalid;
  }
  int64_t major {0}, minor {0}, patch {0};
  try {
    auto j_version = *in.FindMemberByKey("version");
    major = j_version.GetArrayElem(0).GetInt();
    minor = j_version.GetArrayElem(1).GetInt();
    patch = j_version.GetArrayElem(2).GetInt();
  } catch (dmlc::Error const& e) {
    LOG(FATAL) << "Invaid version format in loaded JSON object: ";
  }

  return std::make_tuple(major, minor, patch);
}

Version::TripletT Version::Load(dmlc::Stream* fi) {
  XGBoostVersionT major{0}, minor{0}, patch{0};
  // This is only used in DMatrix serialization, so doesn't break model compability.
  std::string msg { "Incorrect version format found in binary file.  "
                    "Binary file from XGBoost < 1.0.0 is no longer supported. "
                    "Please generate it again." };
  std::string verstr { u8"version:" }, read;
  read.resize(verstr.size(), 0);

  CHECK_EQ(fi->Read(&read[0], verstr.size()), verstr.size()) << msg;
  if (verstr != read) {
    // read might contain `\0` that terminates the string.
    LOG(FATAL) << msg;
  }

  CHECK_EQ(fi->Read(&major, sizeof(major)), sizeof(major)) << msg;
  CHECK_EQ(fi->Read(&minor, sizeof(major)), sizeof(minor)) << msg;
  CHECK_EQ(fi->Read(&patch, sizeof(major)), sizeof(patch)) << msg;

  return std::make_tuple(major, minor, patch);
}

void Version::Save(experimental::Json* out) {
  int64_t major, minor, patch;
  std::tie(major, minor, patch)= Self();
  auto j_version = out->CreateMember("version");
  j_version.SetArray(3);
  j_version.GetArrayElem(0).SetInteger(major);
  j_version.GetArrayElem(1).SetInteger(minor);
  j_version.GetArrayElem(2).SetInteger(patch);
}

void Version::Save(dmlc::Stream* fo) {
  XGBoostVersionT major, minor, patch;
  std::tie(major, minor, patch) = Self();
  std::string verstr { u8"version:" };
  fo->Write(&verstr[0], verstr.size());
  fo->Write(&major, sizeof(major));
  fo->Write(&minor, sizeof(minor));
  fo->Write(&patch, sizeof(patch));
}

std::string Version::String(TripletT const& version) {
  std::stringstream ss;
  ss << std::get<0>(version) << "." << std::get<1>(version) << "." << std::get<2>(version);
  return ss.str();
}

Version::TripletT Version::Self() {
  return std::make_tuple(XGBOOST_VER_MAJOR, XGBOOST_VER_MINOR, XGBOOST_VER_PATCH);
}

bool Version::Same(TripletT const& triplet) {
  return triplet == Self();
}

}  // namespace xgboost
