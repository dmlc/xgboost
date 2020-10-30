/*!
 * Copyright 2019 XGBoost contributors
 */
#include <dmlc/io.h>

#include <string>
#include <tuple>
#include <vector>

#include "xgboost/logging.h"
#include "xgboost/json.h"
#include "xgboost/version_config.h"
#include "version.h"

namespace xgboost {

const Version::TripletT Version::kInvalid {-1, -1, -1};

Version::TripletT Version::Load(Json const& in) {
  if (get<Object const>(in).find("version") == get<Object const>(in).cend()) {
    return kInvalid;
  }
  Integer::Int major {0}, minor {0}, patch {0};
  try {
    auto const& j_version = get<Array const>(in["version"]);
    std::tie(major, minor, patch) = std::make_tuple(
        get<Integer const>(j_version.at(0)),
        get<Integer const>(j_version.at(1)),
        get<Integer const>(j_version.at(2)));
  } catch (dmlc::Error const& e) {
    LOG(FATAL) << "Invaid version format in loaded JSON object: " << in;
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

  CHECK(fi->Read(&major)) << msg;
  CHECK(fi->Read(&minor)) << msg;
  CHECK(fi->Read(&patch)) << msg;

  return std::make_tuple(major, minor, patch);
}

void Version::Save(Json* out) {
  Integer::Int major, minor, patch;
  std::tie(major, minor, patch)= Self();
  (*out)["version"] = std::vector<Json>{Json(Integer{major}),
                                        Json(Integer{minor}),
                                        Json(Integer{patch})};
}

void Version::Save(dmlc::Stream* fo) {
  XGBoostVersionT major, minor, patch;
  std::tie(major, minor, patch) = Self();
  std::string verstr { u8"version:" };
  fo->Write(&verstr[0], verstr.size());
  fo->Write(major);
  fo->Write(minor);
  fo->Write(patch);
}

std::string Version::String(TripletT const& version) {
  std::stringstream ss;
  ss << std::get<0>(version) << "." << get<1>(version) << "." << get<2>(version);
  return ss.str();
}

Version::TripletT Version::Self() {
  return std::make_tuple(XGBOOST_VER_MAJOR, XGBOOST_VER_MINOR, XGBOOST_VER_PATCH);
}

bool Version::Same(TripletT const& triplet) {
  return triplet == Self();
}

}  // namespace xgboost
